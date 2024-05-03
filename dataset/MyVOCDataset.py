import torch
from torchvision.datasets import VOCDetection
import torchvision.transforms as transforms
from PIL import Image

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

category_to_id_mapping = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19,
}


class MyVOCDataset(VOCDetection):
    """
    modify VOCDetection class, change the output type to one-hot vector so that it can be used
    in multi-label classification
    """

    def __init__(self, root: str, year: str, image_set: str, transforms=None, download=False):
        super().__init__(root, year=year, image_set=image_set, transforms=transforms, download=download)

    def __getitem__(self, index: int):
        # copy from VOCDetection
        img = Image.open(self.images[index]).convert("RGB")
        target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())

        # change the label of the image to one-hot vector
        objects = target['annotation']['object']  # get all the objects in this image
        label_list = [objects[i]['name'] for i in range(len(objects))]
        one_hot = torch.zeros(20)  # generate a one-hot vector with size 20
        for label in label_list:
            one_hot[category_to_id_mapping[label]] = 1.0

        # apply transform, here we only need to apply on img
        if self.transforms is not None:
            img = self.transforms(img)

        return img, one_hot


if __name__ == "__main__":
    """
    the code below is for testing the class, ignore it when using MyVOCDataset
    """
    VOC_PATH = "../data/VOC"
    transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor()
    ])
    train_data = MyVOCDataset(VOC_PATH, year='2012', image_set="train", transforms=transforms, download=False)

    a, b = train_data[0]
    print(b)
    print(type(b))
