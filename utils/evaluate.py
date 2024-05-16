import numpy as np
import torch
from torchvision.transforms import transforms
from tqdm import tqdm
import random
import utils.model_process as mp

from model.MLFormer import MLFormer
from model.MyResNet import MyResNet
from model.MyViT import MyViT
from utils.metrics import VOCmAP
from utils.prep import train_one_epoch, evaluate, prep_VOC12
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

img_branch = MyViT(pretrained=True)
img_branch.to(device)
text_branch = MLFormer(start_layer=11)
text_branch.to(device)
# RESNET = MyResNet()
# RESNET.to(device)

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
VOC_PATH = '../data/VOC'

transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((img_size, img_size), scale=(0.7, 1.0)),
    RandAugment(),  # RandAugment
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),  # 转换为张量
])

val_loader = prep_VOC12(transforms=transforms, batch_size=batch_size, image_set='test', VOC_PATH=VOC_PATH, year='2007')

LOAD_MODEL_PATH = "../parameters/MyModel/2007/image_branch_no_pretrained.pth"  # where to load the model parameters
LOAD_MODEL_PATH_2 = "../parameters/MyModel/2007/text_branch_224_start5.pth"  #记得在模型里面也改变参数
# resnet_path = "../parameters/MyModel/ResNet_VOC_asl_augmented.pth"

_, img_branch = mp.load_model(img_branch, LOAD_MODEL_PATH, device)
max_map, text_branch = mp.load_model(text_branch, LOAD_MODEL_PATH_2, device)
print(max_map)
# _, RESNET = mp.load_model(RESNET, resnet_path, device)

'''
evaluate the model 
'''
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
        final_output = output1
        # final_output = RESNET(images)

        val_map.update(targets.cpu().numpy(), final_output.cpu().numpy())

aps = val_map.get_aps()
mAP = np.mean(aps)
print("ap of all categories:", aps)
print("MAP score during evaluation:", mAP)
