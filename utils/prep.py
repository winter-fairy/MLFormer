import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.MyVOCDataset import MyVOCDataset
from utils.metrics import VOCmAP


def prep_VOC12(VOC_PATH, transforms=None, image_set='train', batch_size=32, year='2012'):
    """
    get the dataloader for the specified dataset
    :param VOC_PATH: 数据集存放路径
    :param year: 数据集年份
    :param transforms:
    :param image_set: 'train', 'val' or 'test'
    :param batch_size:
    :return: dataloader
    """
    dataset = MyVOCDataset(root=VOC_PATH,
                           year=year,
                           image_set=image_set,
                           download=False,
                           transforms=transforms)
    if image_set == 'train':
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    调用该函数以训练一个epoch
    :param model:
    :param train_loader:
    :param criterion: 损失函数
    :param optimizer: 优化策略
    :param device:
    :return:
    """
    model.train()
    total_loss = 0
    for _, (images, targets) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        targets = targets.to(device)
        # 前向传播
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Training loss: {avg_loss:.4f}")


def evaluate(model, loader, device):
    """
    evaluate the performance on the validation or test set, mAP will be calculated and printed
    :param model: the ML model
    :param loader: dataloader for validation or test set
    :param device: "cuda" if torch.cuda.is_available() else "cpu"
    :return: mAP
    """
    model.eval()
    val_map = VOCmAP()  # 利用自己定义的mAP类来计算mAP
    val_map.reset()
    with torch.no_grad():
        for _, (images, targets) in enumerate(tqdm(loader)):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            val_map.update(targets.cpu().numpy(), outputs.cpu().numpy())

    aps = val_map.get_aps()
    mAP = np.mean(aps)
    print("MAP score during evaluation:", mAP)
    return mAP