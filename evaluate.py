import torch

def compute_metrics(preds, targets, smooth=1e-6):
    """
    计算 batch 中每张图像的 IoU 和 Dice 系数，并返回平均值
    Args:
        preds (Tensor): 模型预测值，形状为 [B, 1, H, W]
        targets (Tensor): Ground truth 标签，形状为 [B, 1, H, W]
        smooth (float): 防止除以 0 的平滑项
    Returns:
        avg_iou (float): 平均 IoU
        avg_dice (float): 平均 Dice
    """
    batch_size = preds.size(0)
    ious = []
    dices = []

    for i in range(batch_size):
        pred = preds[i].view(-1)
        target = targets[i].view(-1)

        intersection = (pred * target).sum()
        total = pred.sum() + target.sum()
        union = total - intersection

        iou = (intersection + smooth) / (union + smooth)
        dice = (2 * intersection + smooth) / (total + smooth)

        ious.append(iou.item())
        dices.append(dice.item())

    avg_iou = sum(ious) / len(ious)
    avg_dice = sum(dices) / len(dices)

    return avg_iou, avg_dice
