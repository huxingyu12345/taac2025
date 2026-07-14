# main.py
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.amp import GradScaler, autocast

from dataset import MyDataset
from model import BaselineModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)  # 减小 batch_size 防 OOM
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)  # 增加 Transformer 层
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--num_heads', default=8, type=int)  # 增加注意力头
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    # 移除norm_first参数，因为HSTU不需要这个参数
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])
    parser.add_argument('--infonce_alpha', default=1, type=float, help='InfoNCE 损失的权重')
    parser.add_argument('--temperature', default=0.07, type=float, help='InfoNCE 损失的温度参数')
    # HSTU特定参数
    parser.add_argument('--hstu_linear_activation', default='silu', type=str, choices=['silu', 'relu', 'none'], 
                        help='HSTU中的线性激活函数')
    parser.add_argument('--hstu_enable_relative_bias', action='store_true', default=False,
                        help='是否启用HSTU的相对位置偏置')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.99, 0.01])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    scaler = GradScaler('cuda')

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0
    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    triplet_criterion = nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')  # margin=1.0可调
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break

        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            token_type = token_type.to(args.device)  # 确保 token_type 在 cuda
            next_token_type = next_token_type.to(args.device)  # 关键：确保 next_token_type 在 cuda
    
            with autocast('cuda'):
                pos_dist, neg_dist, log_feats, pos_embs, neg_embs = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                )
                indices = torch.nonzero(next_token_type == 1, as_tuple=False)  # indices 自动在 cuda 上
                if len(indices) > 0:
                    # 确保索引计算在 cuda 上
                    index = indices[:, 0] * args.maxlen + indices[:, 1]  # args.maxlen 是 Python int，运算后仍为 cuda 张量
                    anchor = torch.index_select(log_feats.view(-1, args.hidden_units), 0, index)
                    pos_selected = torch.index_select(pos_embs.view(-1, args.hidden_units), 0, index)
                    neg_selected = torch.index_select(neg_embs.view(-1, args.hidden_units), 0, index)
                    triplet_loss = triplet_criterion(anchor, pos_selected, neg_selected)
                else:
                    triplet_loss = torch.tensor(0.0, device=args.device, requires_grad=True)
                
                infonce_loss = model.compute_infonce_loss(log_feats, pos_embs, neg_embs, next_token_type, writer, global_step)
                total_loss = triplet_loss + 0.5 * infonce_loss  # alpha=0.5

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            log_json = json.dumps(
                {
                    'global_step': global_step,
                    'triplet_loss': triplet_loss.item(),
                    'infonce_loss': infonce_loss.item(),
                    'total_loss': total_loss.item(),
                    'epoch': epoch,
                    'time': time.time()
                }
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train_triplet', triplet_loss.item(), global_step)
            writer.add_scalar('Loss/train_infonce', infonce_loss.item(), global_step)
            writer.add_scalar('Loss/train_total', total_loss.item(), global_step)

            global_step += 1

            for param in model.item_emb.parameters():
                total_loss += args.l2_emb * torch.norm(param)

        model.eval()
        valid_loss_sum = 0
        valid_infonce_loss_sum = 0
        for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            token_type = token_type.to(args.device)  # 确保验证时也在 cuda
            next_token_type = next_token_type.to(args.device)  # 关键
    
            with autocast('cuda'):
                pos_dist, neg_dist, log_feats, pos_embs, neg_embs = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                )
                indices = torch.nonzero(next_token_type == 1, as_tuple=False)
                if len(indices) > 0:
                    index = indices[:, 0] * args.maxlen + indices[:, 1]
                    anchor = torch.index_select(log_feats.view(-1, args.hidden_units), 0, index)
                    pos_selected = torch.index_select(pos_embs.view(-1, args.hidden_units), 0, index)
                    neg_selected = torch.index_select(neg_embs.view(-1, args.hidden_units), 0, index)
                    triplet_loss = triplet_criterion(anchor, pos_selected, neg_selected)
                else:
                    triplet_loss = torch.tensor(0.0, device=args.device)
                infonce_loss = model.compute_infonce_loss(log_feats, pos_embs, neg_embs, next_token_type, writer, global_step)
                total_loss = triplet_loss + 0.5 * infonce_loss
                valid_loss_sum += triplet_loss.item()
                valid_infonce_loss_sum += infonce_loss.item()
        valid_loss_sum /= len(valid_loader)
        valid_infonce_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid_bce', valid_loss_sum, global_step)
        writer.add_scalar('Loss/valid_infonce', valid_infonce_loss_sum, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()