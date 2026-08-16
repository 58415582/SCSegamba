'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

from typing import Iterable
import torch
import time
from tqdm import tqdm


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    args=None,
                    logger=None,
                    scaler=None):

    model.train()
    criterion.train()

    amp_enabled = bool(getattr(args, 'amp', False))

    pbar = tqdm(
        total=len(data_loader.dataloader),
        desc="Initial Loss Fused: Pending"
    )

    for i, data in enumerate(data_loader):

        samples = data['image'].to(torch.device(args.device))
        targets = data['label'].to(torch.device(args.device))

        # 清空上一轮梯度
        optimizer.zero_grad()

        # --------------------------------
        # AMP 前向传播
        # 模型使用混合精度
        # --------------------------------
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output = model(samples)

        # --------------------------------
        # Loss 强制使用 FP32 计算
        # 提高 BCE / Dice 的数值稳定性
        # --------------------------------
        loss_final = criterion(
            output.float(),
            targets.float()
        )

        # --------------------------------
        # 检测 NaN / Inf
        # --------------------------------
        if not torch.isfinite(loss_final):
            print(
                f"[WARNING] Non-finite loss detected | "
                f"epoch={epoch} | iter={i} | "
                f"loss={loss_final.item()}"
            )

            if logger is not None:
                logger.warning(
                    f"Non-finite loss detected | "
                    f"epoch={epoch} | iter={i} | "
                    f"loss={loss_final.item()}"
                )

            optimizer.zero_grad()
            continue

        # --------------------------------
        # AMP 反向传播
        # --------------------------------
        if scaler is not None:
            scaler.scale(loss_final).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_final.backward()
            optimizer.step()

        cur_time = time.strftime(
            '%Y_%m_%d_%H:%M:%S',
            time.localtime(time.time())
        )

        loss_final_str = '{:.4f}'.format(loss_final.item())
        lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"time -> {cur_time} | "
            f"Epoch -> {epoch} | "
            f"image_num -> {data['A_paths']} | "
            f"loss final -> {loss_final_str} | "
            f"lr -> {lr}"
        )

        pbar.set_description(
            f"Loss: {loss_final.item():.4f}"
        )
        pbar.update(1)

    pbar.close()