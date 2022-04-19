import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

transform1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(5, (0.05, 0.05)),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


class CelebaBinaryCalssification(Dataset):
    def __init__(self, images, attributes_list, annots, class_name, transform=False):
        self.images = sorted(images)
        self.attributes_list = attributes_list
        self.annots = annots
        self.class_name = class_name
        self.transfom = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        im_name = self.images[idx]
        target = self.annots[im_name.split('/')[-1]][self.attributes_list.index(self.class_name)]
        image = np.zeros((256, 256, 3), dtype = np.uint8)
        shift_x = (256 - 218) // 2 # celeba sizes
        shift_y = (256 - 178) // 2
        image[shift_x: -shift_x, shift_y: -shift_y] = cv2.imread(im_name)
        if self.transfom:
             image = transform1(image).float()
        else:
            image = transform(image).float()
        return image, int((target + 1) // 2)


class CelebaSegmentation(Dataset):
    def __init__(self, images, class_name):
        self.images = sorted(images)
        self.class_name = class_name
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        im_name = self.images[idx]
        num = im_name.split('/')[-1].split('.')[0]
        num = int(num)
        # check
        segm_name = "/root/dmartynov/CelebAMask-HQ/CelebAMask-HQ-mask-anno/" + str(num//2000) + '/' + str(num).zfill(5) + "_" + self.class_name + ".png"
        image = cv2.imread(im_name)
        image = transform(image).float()
        if os.path.isfile(segm_name):
            segm_image = cv2.imread(segm_name, cv2.IMREAD_GRAYSCALE)
        else:
            segm_image = None #TODO remove
        if segm_image is None: # file not found == no mask
            segm_image = torch.zeros((1, 256, 256))
            # print("no mask")
        segm_image = transform(segm_image).float()
        segm_image = (segm_image > 0).float()

        return image, segm_image
