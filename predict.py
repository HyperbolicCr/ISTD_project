import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import MyDataset
from metrics import iou_score, dice_score
from models.ULite import ULite
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

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

def setup_logger(log_dir='logs', log_file='test.log'):
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
    
    logger = logging.getLogger('TestLogger')
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
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

def save_mask(mask_tensor, save_path):
    """
    将预测的mask tensor保存为图像文件。
    
    参数：
        mask_tensor (torch.Tensor): 预测的mask tensor，形状为 [1, H, W]
        save_path (str): 保存路径
    """
    # 将tensor从 [1, H, W] 转换为 [H, W] 并转为numpy
    mask_np = mask_tensor.squeeze(0).cpu().numpy()
    # 将mask转换为0-255的uint8类型
    mask_np = (mask_np * 255).astype(np.uint8)
    # 使用PIL保存图像
    mask_image = Image.fromarray(mask_np)
    mask_image.save(save_path)

def test(model_path, test_images_dir, test_masks_dir, save_masks_dir, device='cuda'):
    """
    测试函数，用于评估模型在测试集上的性能。
    
    参数：
        model_path (str): 模型权重文件路径。
        test_images_dir (str): 测试图像文件夹路径。
        test_masks_dir (str): 测试掩码文件夹路径。
        save_masks_dir (str): 保存预测掩码的文件夹路径。
        device (str): 使用的设备 ('cuda' 或 'cpu')。
    """
    # 设置日志记录器
    logger = setup_logger(log_dir='logs', log_file='test.log')
    logger.info('开始测试过程')
    
    # 配置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 初始化模型并加载权重
    model = ULite().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f'已加载模型权重: {model_path}')
    else:
        logger.error(f'模型权重文件不存在: {model_path}')
        return
    
    model.eval()
    logger.info('模型已设置为评估模式')
    
    # 创建保存预测掩码的目录
    os.makedirs(save_masks_dir, exist_ok=True)
    logger.info(f'预测掩码将保存在: {save_masks_dir}')
    
    # 创建测试数据集和数据加载器
    test_dataset = MyDataset(
        images_dir=test_images_dir,
        masks_dir=test_masks_dir,
        transform=False  # 测试时通常不进行数据增强
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, num_workers=2, shuffle=False
    )
    logger.info('测试数据集和数据加载器已创建')
    
    # 准备保存IoU数值的列表
    iou_results = []
    
    # 获取图像文件名列表
    image_names = sorted(os.listdir(test_images_dir))
    mask_names = sorted(os.listdir(test_masks_dir))
    
    # 确保图像和掩码数量一致
    assert len(image_names) == len(mask_names), "图像和掩码的数量不一致！"
    
    # 定义变换，用于将模型输出转换为图像保存
    # 此处假设数据已经被正确归一化，如果需要其他变换，可根据实际情况调整
    to_pil = transforms.ToPILImage()
    
    # 测试过程
    logger.info('开始测试阶段')
    test_progress = tqdm(test_loader, desc='测试中', leave=False)
    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_progress):
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 应用 sigmoid 并阈值化
            preds = torch.sigmoid(outputs) > 0.5  # [B, 1, H, W]
            preds = preds.float()
            
            # 计算 IoU
            iou = iou_score(preds, masks).item()
            
            # 获取当前图像的名称
            image_name = image_names[idx]
            mask_name = mask_names[idx]
            base_name = os.path.splitext(image_name)[0]
            # 保存预测掩码
            save_path = os.path.join(save_masks_dir, f"{base_name}_pred_mask.png")
            save_mask(preds.squeeze(0), save_path)
            
            # 记录 IoU
            iou_results.append({
                'Image': image_name,
                'IoU': iou
            })
            
            # 更新进度条
            test_progress.set_postfix({
                'IoU': f'{iou:.4f}'
            })
    
    # 保存 IoU 结果到 CSV 文件
    iou_df = pd.DataFrame(iou_results)
    iou_csv_path = os.path.join(save_masks_dir, 'iou_results.csv')
    iou_df.to_csv(iou_csv_path, index=False)
    logger.info(f'IoU 结果已保存到: {iou_csv_path}')
    
    # 计算并记录平均 IoU
    avg_iou = iou_df['IoU'].mean()
    logger.info(f'测试完成！平均 IoU: {avg_iou:.4f}')
    
    # 可选：保存模型在测试集上的详细指标
    # 可以扩展此部分，保存更多指标如 Dice 分数等
    
if __name__ == '__main__':
    # 示例用法
    # 请根据实际情况修改以下路径
    model_path = './weights/ckpt_val_dice_0.5573.pth'  # 替换为实际的模型权重路径
    test_images_dir = './dataset/val/MDvsFA/image/'
    test_masks_dir = './dataset/val/MDvsFA/mask/'
    save_masks_dir = './results/MDvsFA_predicted_masks/'
    
    test(model_path, test_images_dir, test_masks_dir, save_masks_dir, device='cuda')
