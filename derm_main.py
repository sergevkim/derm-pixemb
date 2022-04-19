import os
import pickle
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms.transforms import Resize
from torch.utils.data import Dataset

from datasets import CelebaBinaryCalssification, CelebaSegmentation
from nn_modules import Image2VectorWithCE, Image2VectorWithMSE, Image2Image


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Our main computing device is '{DEVICE}'")


def load_attributes(file_path):
    annots = {}

    with open(file_path, 'r') as f:
        for line in f.read().splitlines()[2:]:
            values = line.split()
            annots[values[0]] = np.array(list(map(float, values[1:])))

    return annots


def load_identity(file_path):
    identity = {}

    with open(file_path, 'r') as f:
        for line in f:
            filename, ident = line.split()
            identity[filename] = int(ident)

    return identity


ANNOTS = load_attributes('/root/dmartynov/celeba/celeba/list_attr_celeba.txt')
IDENTITY = load_identity('/root/dmartynov/celeba/celeba/identity_CelebA.txt')
ATTRIBUTES_LIST = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
        'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
        'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
        'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young']
SEGM_CLASSES = ['cloth', 'hair', 'nose', 'skin', 'l_eye', 'r_eye', 'l_lip', 'u_lip', 'mouth', 'neck']


def get_paths(path, first, last, max_ident=None):
    _, _, filenames = next(os.walk(path))
    images_paths = []

    #TODO remove max_ident condition (it is always None)
    # and then remove IDENTITY global variable

    if max_ident:
        for filename in sorted(filenames):
            if IDENTITY[filename.split('/')[-1]] < max_ident:
                images_paths.append(os.path.join(path, filename))
    else:
        for filename in sorted(filenames):
            images_paths.append(os.path.join(path, filename))

    images_paths = sorted(images_paths)

    return np.stack(images_paths[first:last])


def get_balanced_paths(path, first, class_name, capacity):
    _, _, filenames = next(os.walk(path))

    for filename in filenames:
        if filename[-3:] == 'jpg':
            continue
        else:
            print(path, filename)

    images_paths = []
    quantity_class_one = 0
    quantity_class_two = 0

    for filename in (sorted(filenames))[first:]:
        current_class = ANNOTS[filename][ATTRIBUTES_LIST.index(class_name)]
        # print(filename, current_class)
        if current_class < 0 and quantity_class_one < capacity // 2:
            images_paths.append(path + filename)
            quantity_class_one += 1
        elif current_class > 0 and quantity_class_two < capacity // 2:
            images_paths.append(path + filename)
            quantity_class_two += 1
        if quantity_class_one + quantity_class_two == capacity:
            break

    return images_paths, int(filename.split('.')[0]) + 1


def main():
    DATASET_QUANTITY = 32
    DATASET_SIZE = 256
    train_datasets = [None] * DATASET_QUANTITY
    test_datasets = [None] * DATASET_QUANTITY
    nn_modules = [None] * DATASET_QUANTITY

    # pretrained_model = Image2VectorWithCE(2)
    # pretrained_model.load_state_dict(torch.load("train_32_64/nn_module_32_0.pt"))

    # 32 - for validation, backbone freeze
    first_image = 0
    for i in range(DATASET_QUANTITY):
        if i < DATASET_QUANTITY * 3 // 4:
            train_images, first_image = get_balanced_paths(
                path='/root/dmartynov/celeba/celeba/img_align_celeba/',
                first=first_image,
                class_name=ATTRIBUTES_LIST[i],
                capacity=DATASET_SIZE,
            )
            test_images, first_image = get_balanced_paths(
                path='/root/dmartynov/celeba/celeba/img_align_celeba/',
                first=first_image,
                class_name=ATTRIBUTES_LIST[i],
                capacity=DATASET_SIZE,
            )
            # train_images = get_paths('/root/dmartynov/celeba/celeba/img_align_celeba/', 2 * i * DATASET_SIZE, (2 * i + 1) * DATASET_SIZE)
            # test_images = get_paths('/root/dmartynov/celeba/celeba/img_align_celeba/', (2 * i + 1) * DATASET_SIZE, (2 * i + 2) * DATASET_SIZE)
            train_datasets[i] = CelebaBinaryCalssification(
                train_images,
                ATTRIBUTES_LIST,
                ANNOTS,
                ATTRIBUTES_LIST[i],
            )
            test_datasets[i] = CelebaBinaryCalssification(
                test_images,
                ATTRIBUTES_LIST,
                ANNOTS,
                ATTRIBUTES_LIST[i],
            )
            nn_modules[i] = Image2VectorWithCE(2)
            # nn_modules[i - 32].encoder = pretrained_model.encoder
        else:
            j = i - DATASET_QUANTITY * 3 // 4 # - 32 + DATASET_QUANTITY
            train_images = get_paths(
                '/root/dmartynov/CelebAMask-HQ/CelebA-HQ-img/',
                2 * j * DATASET_SIZE,
                (2 * j + 1) * DATASET_SIZE,
            )
            test_images = get_paths(
                '/root/dmartynov/CelebAMask-HQ/CelebA-HQ-img/',
                (2 * j + 1) * DATASET_SIZE,
                (2 * j + 2) * DATASET_SIZE,
            )
            train_datasets[i] = CelebaSegmentation(train_images, SEGM_CLASSES[j])
            test_datasets[i] = CelebaSegmentation(test_images, SEGM_CLASSES[j])
            nn_modules[i] = Image2Image()
            # nn_modules[i - 32].encoder = pretrained_model.encoder

    for i in range(DATASET_QUANTITY):
        nn_modules[i].to(DEVICE)

    batch_size = 8
    train_batch_gens = [
        torch.utils.data.DataLoader(
            train_datasets[i],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        ) for i in range(DATASET_QUANTITY)
    ]

    val_batch_gens = [
        torch.utils.data.DataLoader(
            test_datasets[i],
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        ) for i in range(DATASET_QUANTITY)
    ]

    # was weight_decay=3e-5
    optimizers = [
        torch.optim.Adam(
            nn_modules[i].parameters(),
            lr=1e-3,
            weight_decay=3e-4,
        ) for i in range(DATASET_QUANTITY)
    ]

    NUM_EPOCHS = 20
    train_loss = [list() for _ in range(DATASET_QUANTITY)]
    results = [list() for _ in range(DATASET_QUANTITY)]
    for epoch in range(NUM_EPOCHS):
        print("Epoch", len(results[0]))
        for i in range(DATASET_QUANTITY):
            nn_modules[i].train(True)

        for batch in tqdm.tqdm(zip(*train_batch_gens)):
            for i in range(DATASET_QUANTITY):
                optimizers[i].zero_grad()

                X_batch = batch[i][0].to(DEVICE)
                y_batch = batch[i][1].to(DEVICE)

                predictions = nn_modules[i](X_batch)
                loss = nn_modules[i].compute_loss(predictions,y_batch)
                train_loss[i].append(loss.cpu().data.numpy())

                loss.backward()
                optimizers[i].step()

        print("train loss")
        for i in range(DATASET_QUANTITY):
            print(i, np.mean(train_loss[i]))
            results[i].append(list())
            results[i][-1].append(np.mean(train_loss[i]))
            train_loss[i] = []

        print("validation loss")
        for i in range(DATASET_QUANTITY):
            nn_modules[i].train(False)
            with torch.no_grad():
                for X_batch, y_batch in val_batch_gens[i]:
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    predictions = nn_modules[i](X_batch)
                    loss = nn_modules[i].compute_loss(predictions,y_batch)
                    train_loss[i].append(loss.cpu().data.numpy())
        for i in range(DATASET_QUANTITY):
            print(i, np.mean(train_loss[i]))
            results[i][-1].append(np.mean(train_loss[i]))
            train_loss[i] = []

        print("train metric")
        for i in range(DATASET_QUANTITY):
            nn_modules[i].train(False)
            metric = []
            with torch.no_grad():
                for X_batch, y_batch in train_batch_gens[i]:
                    X_batch = X_batch.to(DEVICE)
                    y_pred = nn_modules[i](X_batch)
                    y_pred = nn_modules[i].post_processing(y_pred)
                    metric.append(nn_modules[i].metric(y_batch, y_pred.cpu()))
            print(i, np.mean(metric))
            results[i][-1].append(np.mean(metric))

        print("validation metric")
        for i in range(DATASET_QUANTITY):
            nn_modules[i].train(False)
            metric = []
            with torch.no_grad():
                for X_batch, y_batch in val_batch_gens[i]:
                    X_batch = X_batch.to(DEVICE)
                    y_pred = nn_modules[i](X_batch)
                    y_pred = nn_modules[i].post_processing(y_pred)
                    metric.append(nn_modules[i].metric(y_batch, y_pred.cpu()))
            print(i, np.mean(metric))
            results[i][-1].append(np.mean(metric))

        with open('results_train_' + str(DATASET_QUANTITY) + '.pkl', 'wb') as f:
            pickle.dump(results, f)

        Path("/root/sergevkim/checkpoints").mkdir(parents=True, exist_ok=True)
        for i in range(DATASET_QUANTITY):
            # TODO add git commit hash to paths
            model_path = f"checkpoints/nn_module_train_{str(DATASET_QUANTITY)}_{i}.pt"
            optimizer_path = f"checkpoints/optimizer{str(DATASET_QUANTITY)}_{i}.pt"
            torch.save(nn_modules[i].state_dict(), model_path)
            torch.save(optimizers[i].state_dict(), optimizer_path)


if __name__ == '__main__':
    main()
