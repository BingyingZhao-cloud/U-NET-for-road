import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from dataset import RoadSegmentationDataset
from modules import UNet
import config

print(torch.cuda.is_available())      
print(torch.cuda.get_device_name(0))

def get_loaders():
    # 全部训练集
    full_train_dataset = RoadSegmentationDataset(
        image_dir=config.TRAIN_IMAGE_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
    )
    total_samples = len(full_train_dataset)
    val_train_size = int(0.2 * total_samples)
    train_size = total_samples - val_train_size

    train_subset, valid_subset = random_split(full_train_dataset, [train_size, val_train_size])

    train_loader = DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    valid_loader = DataLoader(
        valid_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    return train_loader, valid_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, masks) in enumerate(loader, 1):
        images = images.to(device)
        masks = masks.to(device) if masks is not None else None

        optimizer.zero_grad()
        outputs = model(images)

        # 仅计算那些具有标签样本的 loss
        if masks is not None and torch.any(masks):
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if batch_idx % config.LOG_INTERVAL == 0:
            avg = running_loss / config.LOG_INTERVAL if running_loss != 0 else 0.0
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {avg:.4f}")
            running_loss = 0.0


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    count = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            # 如果没有 mask，跳过 loss 计算，但可以进行预测显示
            if masks is None or masks.sum() == 0:
                continue
            masks = masks.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, masks).item()
            count += 1

    if count > 0:
        return val_loss / count
    else:
        return 0

def main():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 获取 DataLoader：训练集和验证集（结合了 train 部分和 valid 文件夹数据）
    train_loader, valid_loader = get_loaders()

    # 构建模型、损失、优化器
    model = UNet(
        in_channels=config.INPUT_CHANNELS,
        out_channels=config.OUTPUT_CHANNELS,
        bilinear=True
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_loss = float('inf')
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, valid_loader, criterion, device)
        print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME)
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path}")

if __name__ == '__main__':
    main()
