import numpy as np
import torch
from torch import nn, optim
from torchvision.transforms import transforms
from tqdm import tqdm
import random
from utils.aslloss import AsymmetricLossOptimized

import utils.model_process as mp
from model.MyResNet import MyResNet

from model.MLFormer import MLFormer
from model.MyViT import MyViT
from utils.metrics import VOCmAP
from utils.prep import train_one_epoch, evaluate, prep_VOC12
from torchvision.transforms import RandomHorizontalFlip, RandomResizedCrop, RandomErasing
from randaugment import RandAugment

"""
preparation for the model
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 20204146  # 设置随机种子
random.seed(seed)  # 设置Python内置随机数生成器的种子
np.random.seed(seed)  # 设置NumPy随机数生成器的种子
torch.manual_seed(seed)  # 设置PyTorch随机数生成器的种子
# 如果使用CUDA，则还需要设置CUDA的种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 确保每次结果相同
    torch.backends.cudnn.benchmark = False  # 禁用CuDNN的确定性算法，以获得可重复的结果

img_branch = MyViT()
img_branch.to(device)
text_branch = MLFormer()
text_branch.to(device)

res = []


def forward_hook(module, input, output):
    res.append(output)


# 为img_branch注册hook函数，获取每一层ViT的输出
for _, model in img_branch.named_children():
    for name1, module in model.named_children():
        if name1 == 'blocks':
            for name2, block in module.named_children():
                block.register_forward_hook(forward_hook)

batch_size = 16
img_size = 224
learning_rate = 0.000001
num_epochs = 60
VOC_PATH = '../data/VOC'

criterion = AsymmetricLossOptimized()  # 二进制交叉熵损失
optimizer = optim.Adam(text_branch.parameters(), lr=learning_rate, betas=(0.999, 0.9), weight_decay=1e-2)  # Adam优化器

transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((img_size, img_size), scale=(0.7, 1.0)),
    RandAugment(),  # RandAugment
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),  # 转换为张量
])

train_loader = prep_VOC12(transforms=transforms, batch_size=batch_size, image_set='train', VOC_PATH=VOC_PATH)
val_loader = prep_VOC12(transforms=transforms, batch_size=batch_size, image_set='val', VOC_PATH=VOC_PATH)

"""
train the model
"""
max_mAP = 0.0  # record the max mAP during training
SAVE_MODEL_PATH = "../parameters/MyModel/text_branch_224.pth"  # where to save the model parameters
LOAD_MODEL_PATH = "../parameters/MyModel/image_branch_224.pth"  # where to load the model parameters
LOAD_MODEL_PATH_2 = "../parameters/MyModel/text_branch_224.pth"

_, img_branch = mp.load_model(img_branch, LOAD_MODEL_PATH, device)
max_mAP, text_branch = mp.load_model(text_branch, LOAD_MODEL_PATH_2, device)
print("current best mAP:", max_mAP)

early_stopping = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # 训练一个epoch
    text_branch.train()
    total_loss = 0.0
    for _, (images, targets) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        targets = targets.to(device)
        # 前向传播
        res.clear()
        output1 = img_branch(images)  # 经过图像分支，得到第一个置信度
        output2 = text_branch(res)  # 经过标签分支
        final_output = output2
        # 计算损失
        loss = criterion(final_output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    epoch_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # 每训练一个epoch, 在验证集上进行评估
    text_branch.eval()
    val_map = VOCmAP()  # 利用自己定义的mAP类来计算mAP
    val_map.reset()
    with torch.no_grad():
        for _, (images, targets) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            targets = targets.to(device)

            res.clear()
            output1 = img_branch(images)  # 经过图像分支，得到第一个置信度
            output2 = text_branch(res)
            final_output = output2

            val_map.update(targets.cpu().numpy(), final_output.cpu().numpy())

    aps = val_map.get_aps()
    mAP = np.mean(aps)
    print("MAP score during evaluation:", mAP)
    if mAP > max_mAP:
        mp.save_model(text_branch, mAP, SAVE_MODEL_PATH)
        max_mAP = mAP
        early_stopping = 0
    else:
        early_stopping = early_stopping + 1
        if early_stopping >= 60:
            print("Early stop")
            break
