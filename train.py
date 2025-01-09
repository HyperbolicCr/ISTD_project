import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import MyDataset
from metrics import iou_score, dice_score, DiceLoss
from models.ULite import ULite
from torch.utils.tensorboard import SummaryWriter
import logging
from datetime import datetime
import numpy as np  

# 定义 F1 分数计算函数
def f1_score_fn(outputs, masks, threshold=0.5):
    """
    计算 F1 分数
    
    参数：
        outputs (torch.Tensor): 模型输出，形状为 [B, 1, H, W]
        masks (torch.Tensor): 真实掩码，形状为 [B, 1, H, W]
        threshold (float): 阈值，用于将输出概率转换为二值预测
    
    返回：
        float: F1 分数
    """
    preds = (outputs > threshold).float()
    masks = masks.float()
    
    tp = (preds * masks).sum(dim=(1, 2, 3))
    fp = (preds * (1 - masks)).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * masks).sum(dim=(1, 2, 3))
    
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)  # 加一个小常数避免除零
    return f1.mean().item()

def setup_logger(log_dir='logs', log_file='training.log'):
    """
    设置日志记录器，记录到控制台和文件中。
    
    参数：
        log_dir (str): 日志文件夹路径。
        log_file (str): 日志文件名。
    
    返回：
        logger (logging.Logger): 配置好的日志记录器。
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)
    
    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 文件处理器
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def mixup_data(x, y, alpha=1.0, device='cpu'):
    """
    生成混合后的输入和标签
    
    参数：
        x (torch.Tensor): 输入图像, 形状 [B, C, H, W]
        y (torch.Tensor): 掩码标签, 形状 [B, 1, H, W]
        alpha (float): Beta分布的参数，alpha>0 越大，混合越平均
        device (str): 当前使用的设备
        
    返回：
        mixed_x (torch.Tensor): 混合后的图像
        y_a (torch.Tensor): 原标签
        y_b (torch.Tensor): 随机打乱后的标签
        lam (float): 混合系数
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    # 随机打乱索引
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index, :]
    return mixed_x, y_a, y_b, lam

def train(use_mixup=False, mixup_alpha=1.0):
    """
    训练函数
    
    参数：
        use_mixup (bool): 是否使用MixUp
        mixup_alpha (float): mixup的alpha参数
    """
    # 设置日志记录器
    logger = setup_logger(log_dir='logs', log_file='training_SIRST_mixup.log')
    logger.info('开始训练过程')

    # 配置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')

    # 初始化模型并移动到设备
    model = ULite().to(device)
    logger.info('模型已初始化并移动到设备')

    # 定义损失函数
    criterion = DiceLoss()
    logger.info('损失函数已定义')

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=1.6e-2)
    logger.info('优化器已定义')

    # 定义学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    logger.info('学习率调度器已定义')

    # 初始化 TensorBoard 写入器
    log_dir = os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f'TensorBoard 写入器已初始化，日志目录: {log_dir}')

    # 创建数据集和数据加载器
    train_dataset = MyDataset(
        images_dir='./dataset/training/image/',
        masks_dir='./dataset/training/mask/',
        transform=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=64, num_workers=8, shuffle=True
    )
    logger.info('训练数据集和数据加载器已创建')

    val_dataset = MyDataset(
        images_dir='./dataset/val/SIRST/image/',
        masks_dir='./dataset/val/SIRST/mask/',
        transform=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, num_workers=2, shuffle=False
    )
    logger.info('验证数据集和数据加载器已创建')

    # 创建保存模型权重的目录
    os.makedirs('weights', exist_ok=True)
    checkpoint_dir = 'weights'
    logger.info(f'模型权重将保存在: {checkpoint_dir}')

    # 训练配置
    max_epochs = 20
    best_val_dice = -1  # 初始化最佳验证 Dice 分数

    for epoch in range(1, max_epochs + 1):
        logger.info(f'\nEpoch {epoch}/{max_epochs}')
        logger.info('-' * 30)

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_iou = 0.0
        train_steps = 0

        train_progress = tqdm(train_loader, desc='训练中', leave=False)
        for images, masks in train_progress:
            images = images.to(device)
            masks = masks.to(device)

            # 如果启用MixUp，则进行混合
            if use_mixup:
                images, masks_a, masks_b, lam = mixup_data(
                    images, masks, alpha=mixup_alpha, device=device
                )
            else:
                lam = 1.0
                masks_a, masks_b = masks, masks

            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)
            if use_mixup:
                # 使用混合后的标签计算损失
                loss = lam * criterion(outputs, masks_a) + (1 - lam) * criterion(outputs, masks_b)
            else:
                loss = criterion(outputs, masks)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 计算指标（对MixUp情况，指标可使用混合前的mask_a或原mask查看）
            dice = dice_score(outputs, masks_a).item()  # 这里可任选masks_a、masks_b或原masks
            iou = iou_score(outputs, masks_a).item()

            train_loss += loss.item()
            train_dice += dice
            train_iou += iou
            train_steps += 1

            # 更新进度条
            train_progress.set_postfix({
                'Loss': f'{train_loss / train_steps:.4f}',
                'Dice': f'{train_dice / train_steps:.4f}',
                'IoU': f'{train_iou / train_steps:.4f}'
            })

        avg_train_loss = train_loss / train_steps
        avg_train_dice = train_dice / train_steps
        avg_train_iou = train_iou / train_steps

        logger.info(f'训练 - Loss: {avg_train_loss:.4f} | Dice: {avg_train_dice:.4f} | IoU: {avg_train_iou:.4f}')

        # 记录训练指标到 TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Dice/train', avg_train_dice, epoch)
        writer.add_scalar('IoU/train', avg_train_iou, epoch)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        val_f1 = 0.0
        val_steps = 0

        val_progress = tqdm(val_loader, desc='验证中', leave=False)
        with torch.no_grad():
            for images, masks in val_progress:
                images = images.to(device)
                masks = masks.to(device)

                # 前向传播
                try:
                    outputs = model(images)
                except Exception as e:
                    logger.error(f'Error processing images: {e}')
                    continue
                loss = criterion(outputs, masks)

                # 计算指标
                dice = dice_score(outputs, masks).item()
                iou = iou_score(outputs, masks).item()
                f1 = f1_score_fn(outputs, masks)

                val_loss += loss.item()
                val_dice += dice
                val_iou += iou
                val_f1 += f1
                val_steps += 1

                # 更新进度条
                val_progress.set_postfix({
                    'Loss': f'{val_loss / val_steps:.4f}',
                    'Dice': f'{val_dice / val_steps:.4f}',
                    'IoU': f'{val_iou / val_steps:.4f}',
                    'F1': f'{val_f1 / val_steps:.4f}'
                })

        avg_val_loss = val_loss / val_steps
        avg_val_dice = val_dice / val_steps
        avg_val_iou = val_iou / val_steps
        avg_val_f1 = val_f1 / val_steps

        logger.info(f'验证 - Loss: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f} | IoU: {avg_val_iou:.4f} | F1: {avg_val_f1:.4f}')

        # 记录验证指标到 TensorBoard
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Dice/val', avg_val_dice, epoch)
        writer.add_scalar('IoU/val', avg_val_iou, epoch)
        writer.add_scalar('F1/val', avg_val_f1, epoch)

        # 更新学习率调度器
        scheduler.step(avg_val_dice)
        logger.info('学习率调度器已更新')

        # 检查是否为最佳模型，若是则保存
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_val_dice_{best_val_dice:.4f}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f'✅ 新的最佳模型已保存: {checkpoint_path}')
            # 记录最佳模型路径到 TensorBoard
            writer.add_text('Best Model', checkpoint_path, epoch)

    # 打印训练完成信息
    logger.info('\n训练完成!')
    logger.info(f'最佳验证 Dice 分数: {best_val_dice:.4f}')

    # 关闭 TensorBoard 写入器
    writer.close()

if __name__ == '__main__':
    train(use_mixup=True, mixup_alpha=1)