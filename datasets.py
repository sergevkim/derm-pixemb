from pathlib import Path
import os
import typing as tp

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

ATTRIBUTES_LIST = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie',
    'Young',
]

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
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im_name = self.images[idx]
        target = self.annots[im_name.split('/')[-1]][self.attributes_list.index(self.class_name)]
        image = np.zeros((256, 256, 3), dtype = np.uint8)
        shift_x = (256 - 218) // 2 # celeba sizes
        shift_y = (256 - 178) // 2
        image[shift_x: -shift_x, shift_y: -shift_y] = cv2.imread(im_name)

        if self.transform:
            image = transform1(image).float()
        else:
            image = transform(image).float()

        return image, int((target + 1) // 2)


def load_attributes(file_path):
    annots = {}

    with open(file_path, 'r') as f:
        for line in f.read().splitlines()[2:]:
            values = line.split()
            annots[values[0]] = np.array(list(map(float, values[1:])))

    return annots


def prepare_pos_neg_paths(path2labels, class_index: int = 5):  # 5 is Bald
    pos_paths = list()
    neg_paths = list()

    for path, labels in path2labels.items():
        if labels[class_index] == 1:
            pos_paths.append(path)
        else:
            neg_paths.append(path)

    return pos_paths, neg_paths


class CelebAPosNegDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        class_name: str,
        transform,
        n_pairs: int,
        pos_left_border: tp.Optional[int] = None,
        pos_right_border: tp.Optional[int] = None,
        neg_left_border: tp.Optional[int] = None,
        neg_right_border: tp.Optional[int] = None,
    ):
        self.data_path = data_path
        self.class_name = class_name
        self.transform = transform
        self.n_pairs = n_pairs

        self.images_dir_path = data_path / 'img_align_celeba'
        self.class_index = ATTRIBUTES_LIST.index(class_name)
        self.path2labels = load_attributes(data_path / 'list_attr_celeba.txt')
        self.pos_paths, self.neg_paths = \
            prepare_pos_neg_paths(self.path2labels, self.class_index)
        if pos_left_border is not None:
            self.pos_paths = self.pos_paths[pos_left_border:pos_right_border]
            self.neg_paths = self.neg_paths[neg_left_border:neg_right_border]

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        pos_label = torch.tensor(1).float()  # we push pos_image_prob be more than neg_image_prob always
        neg_label = torch.tensor(0).float()
        print('paths lengths:', len(self.pos_paths), len(self.neg_paths))
        pos_path = self.images_dir_path / np.random.choice(self.pos_paths)
        neg_path = self.images_dir_path / np.random.choice(self.neg_paths)

        pos_image = cv2.imread(str(pos_path))
        neg_image = cv2.imread(str(neg_path))

        transformed_pos_image = self.transform(pos_image)
        transformed_neg_image = self.transform(neg_image)

        return transformed_pos_image, transformed_neg_image, pos_label, neg_label


class CelebAPosNegDatasetV2(Dataset):
    def __init__(
        self,
        data_path: Path,
        class_name: str,
        transform,
        n_pairs: int,

        first: int,
        minimal_pos_number: int = 250,
        minimal_neg_number: int = 250,
    ):
        self.data_path = data_path
        self.class_name = class_name
        self.transform = transform
        self.n_pairs = n_pairs

        self.images_dir_path = data_path / 'img_align_celeba'
        self.class_index = ATTRIBUTES_LIST.index(class_name)
        self.path2labels = load_attributes(data_path / 'list_attr_celeba.txt')
        self.pos_paths = list()
        self.neg_paths = list()

        filenames = \
            sorted([p.name for p in self.images_dir_path.glob('*.jpg')])
        for i in range(first, len(filenames)):
            filename = filenames[i]
            if self.path2labels[filename][self.class_index] == 1:
                self.pos_paths.append(filename)
            else:
                self.neg_paths.append(filename)

            if (
                len(self.pos_paths) >= minimal_pos_number and
                len(self.neg_paths) >= minimal_neg_number
            ):
                self.last_image_idx = i+1
                break

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        pos_label = torch.tensor(1).float()  # we push pos_image_prob be more than neg_image_prob always
        neg_label = torch.tensor(0).float()
        pos_path = self.images_dir_path / np.random.choice(self.pos_paths)
        neg_path = self.images_dir_path / np.random.choice(self.neg_paths)

        pos_image = cv2.imread(str(pos_path))
        neg_image = cv2.imread(str(neg_path))

        transformed_pos_image = self.transform(pos_image)
        transformed_neg_image = self.transform(neg_image)

        return transformed_pos_image, transformed_neg_image, pos_label, neg_label


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
        # check; TODO path to a data directory
        segm_name = "/root/dmartynov/CelebAMask-HQ/CelebAMask-HQ-mask-anno/" + str(num//2000) + '/' + str(num).zfill(5) + "_" + self.class_name + ".png"
        image = cv2.imread(im_name)
        image = transform(image).float()

        # for some images we have no segm mask - so we need do filter them out
        if os.path.isfile(segm_name):
            segm_image = cv2.imread(segm_name, cv2.IMREAD_GRAYSCALE)
        else:
            segm_image = torch.zeros((1, 256, 256))
            # print("no mask")

        segm_image = transform(segm_image).float()
        segm_image = (segm_image > 0).float()

        return image, segm_image
