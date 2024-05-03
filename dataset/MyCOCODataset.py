import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms

id_index = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14,
            17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27,
            33: 28, 34: 29,
            35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42,
            49: 43, 50: 44,
            51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57,
            64: 58, 65: 59,
            67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72,
            84: 73, 85: 74,
            86: 75, 87: 76, 88: 77, 89: 78, 90: 79}


class MyCOCODataset(CocoDetection):
    def __init__(self, root, annFile, transforms):
        super().__init__(root=root, annFile=annFile, transforms=transforms)

    def __getitem__(self, index):
        # copy from CocoDetection and slightly modified
        Id = self.ids[index]
        image = self._load_image(Id)
        target = self._load_target(Id)

        if self.transforms is not None:
            image = self.transforms(image)

        # make change so that this class can be used for multilabel classification
        ids = [tmp['category_id'] for tmp in target]  # 取到图中所有物品对应的ID
        # 把ID转换为one hot vector
        one_hot = torch.zeros(80)
        for ID in ids:
            one_hot[id_index[ID]] = 1.0  # 把ID对应的index位置变为1

        return image, one_hot


if __name__ == '__main__':
    ANN_PATH = '../data/MSCOCO/annotations/instances_train2017.json'
    ROOT_PATH = '../data/MSCOCO/train2017'
    my_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    my_coco = MyCOCODataset(root=ROOT_PATH, annFile=ANN_PATH, transforms=my_transforms)

    a,b = my_coco[0]  # a为(3,224,224)的张量，b为大小为80的张量，表示这个图片对应的one-hot标签

