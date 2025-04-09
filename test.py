import os
import torch
from torch.utils.data import DataLoader
from dataset import RoadSegmentationDataset
from modules import UNet
from evaluate import compute_metrics
from config import TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, IMAGE_SIZE, INPUT_CHANNELS, OUTPUT_CHANNELS, CHECKPOINT_DIR, MODEL_NAME, BINARIZATION_THRESHOLD
from transforms import JointCompose, Resize, ToTensor

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = UNet(in_channels=INPUT_CHANNELS, out_channels=OUTPUT_CHANNELS)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, MODEL_NAME), map_location=device))
    model.to(device)
    model.eval()

    # 数据变换
    test_transform = JointCompose([
        Resize(IMAGE_SIZE),
        ToTensor()
    ])

    # 使用训练集划分出来的验证数据部分（有mask）
    val_dataset = RoadSegmentationDataset(
        image_dir=TRAIN_IMAGE_DIR,
        mask_dir=TRAIN_MASK_DIR,
        transform=test_transform,
        threshold=BINARIZATION_THRESHOLD
    )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 评估模型
    ious = []
    dices = []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            iou, dice = compute_metrics(preds, masks)
            ious.append(iou)
            dices.append(dice)

    mean_iou = torch.tensor(ious).mean().item()
    mean_dice = torch.tensor(dices).mean().item()
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")

if __name__ == '__main__':
    test()
