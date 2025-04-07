import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import RoadSegmentationDataset
from modules import UNet
import config

print(torch.cuda.is_available())        # 应该输出 True
print(torch.cuda.get_device_name(0))

def get_loaders():
    train_ds = RoadSegmentationDataset(
        image_dir=config.TRAIN_IMAGE_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
    )
    val_ds = RoadSegmentationDataset(
        image_dir=config.VALID_IMAGE_DIR,
        mask_dir=config.VALID_MASK_DIR,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    return train_loader, val_loader

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, masks) in enumerate(loader, 1):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % config.LOG_INTERVAL == 0:
            avg = running_loss / config.LOG_INTERVAL
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {avg:.4f}")
            running_loss = 0.0

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, masks).item()
    return val_loss / len(loader)

def main():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_loader, val_loader = get_loaders()

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
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME)
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path}")

if __name__ == '__main__':
    main()
