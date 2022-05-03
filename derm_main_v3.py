import os
import pickle
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Resize

from datasets import (CelebaBinaryCalssification, CelebAPosNegDataset,
                      CelebAPosNegDatasetV2, CelebaSegmentation)
from nn_modules import (Image2Image, Image2VectorWithCE,
                        Image2VectorWithMSE, Image2VectorWithCE_Pairwise)


def get_computing_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

device = get_computing_device()
print(f"Our main computing device is '{device}'")

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

def load_landmarks(file_path):
    landmarks = {}
    with open(file_path, 'r') as f:
        for line in f.read().splitlines()[2:]:
            values = line.split()
            landmarks[values[0]] = np.array(list(map(float, values[1:])))
    return landmarks

annots = load_attributes('/root/dmartynov/celeba/celeba/list_attr_celeba.txt')
identity = load_identity('/root/dmartynov/celeba/celeba/identity_CelebA.txt')
attributes_list = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
        'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
        'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
        'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young']
landmarks = load_landmarks("/root/dmartynov/celeba/celeba/list_landmarks_align_celeba.txt")
segm_classes = ['cloth', 'hair', 'mouth', 'neck', 'nose', 'skin', 'l_eye', 'r_eye', 'l_lip', 'u_lip']


def get_paths(path, first, last, max_ident=None):
    _, _, filenames = next(os.walk(path))

    images_paths = []
    if max_ident:
        for filename in sorted(filenames):
            if identity[filename.split('/')[-1]] < max_ident:
                images_paths.append(os.path.join(path, filename))
    else:
        for filename in sorted(filenames):
            images_paths.append(os.path.join(path, filename))
    images_paths = sorted(images_paths)
    # print(len(images_paths), first, last)
    return np.stack(images_paths[first:last])

def get_balanced_paths(path, first, class_name, capacity):
    _, _, filenames = next(os.walk(path))
    images_paths = []
    quantity_class_one = 0
    quantity_class_two = 0
    for filename in (sorted(filenames))[first:]:
        current_class = annots[filename][attributes_list.index(class_name)]
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


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

WIDTH, HEIGHT = 178, 218
posneg_dataset_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(((256 - WIDTH) // 2, (256 - HEIGHT) // 2)),
    transforms.ToTensor()
])

DATASET_QUANTITY = 32
DATASET_SIZE = 5000
train_datasets = [None] * DATASET_QUANTITY
test_datasets = [None] * DATASET_QUANTITY
nn_modules = [None] * DATASET_QUANTITY


# pretrained_model = Image2VectorWithCE(2)
# pretrained_model.load_state_dict(torch.load("train_32_64/nn_module_32_0.pt"))

# 32 - for validation, backbone freeze
first_image = 0
for i in range(DATASET_QUANTITY):
    if i < DATASET_QUANTITY * 3 // 4:
        #train_images, first_image = get_balanced_paths('/root/dmartynov/celeba/celeba/img_align_celeba/', first_image, attributes_list[i], 5000)
        print(first_image)
        if first_image > 120_000:
            first_image = 0
        # train_images = get_paths('/root/dmartynov/celeba/celeba/img_align_celeba/', 2 * i * DATASET_SIZE, (2 * i + 1) * DATASET_SIZE)
        #test_images, _ = get_balanced_paths('/root/dmartynov/celeba/celeba/img_align_celeba/', 160_000, attributes_list[i], 1000)

        # test_images = get_paths('/root/dmartynov/celeba/celeba/img_align_celeba/', (2 * i + 1) * DATASET_SIZE, (2 * i + 2) * DATASET_SIZE)
        # train_datasets[i] = CelebaBinaryCalssification(train_images, attributes_list, annots, attributes_list[i])
        # test_datasets[i] = CelebaBinaryCalssification(test_images, attributes_list, annots, attributes_list[i])
        train_datasets[i] = CelebAPosNegDatasetV2(
            data_path=Path('/root/dmartynov/celeba/celeba'),
            class_name=attributes_list[i],
            transform=posneg_dataset_transform,
            n_pairs=250,
            first=first_image,
            minimal_pos_number=250,
            minimal_neg_number=250,
        )
        first_image = train_datasets[i].last_image_idx
        test_datasets[i] = CelebAPosNegDatasetV2(
            data_path=Path('/root/dmartynov/celeba/celeba'),
            class_name=attributes_list[i],
            transform=posneg_dataset_transform,
            n_pairs=250,
            first=160_000,
            minimal_pos_number=1000,
            minimal_neg_number=1000,
        )
        nn_modules[i] = Image2VectorWithCE_Pairwise(2)
        # nn_modules[i - 32].encoder = pretrained_model.encoder
    else:
        j = i - DATASET_QUANTITY * 3 // 4 # - 32 + DATASET_QUANTITY
        # print(j)
        train_images = get_paths('/root/dmartynov/CelebAMask-HQ/CelebA-HQ-img/', (j % 5) * 5000, (j % 5) * 5000 + 5000)
        test_images = get_paths('/root/dmartynov/CelebAMask-HQ/CelebA-HQ-img/', 29000, 30000)
        train_datasets[i] = CelebaSegmentation(train_images, segm_classes[j])
        test_datasets[i] = CelebaSegmentation(test_images, segm_classes[j])
        nn_modules[i] = Image2Image()
        # nn_modules[i - 32].encoder = pretrained_model.encoder

for i in range(DATASET_QUANTITY):
    print (f"len train_dataset {i} - {len(train_datasets[i])}")
    print (f"len test_dataset {i} - {len(test_datasets[i])}")

for i in range(DATASET_QUANTITY):
    # nn_modules[i].load_state_dict(torch.load("nn_module_train_32_" + str(i) + ".pt"))
    nn_modules[i].to(device)

batch_size = 4
train_batch_gens = [torch.utils.data.DataLoader(train_datasets[i],
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4) for i in range(DATASET_QUANTITY)]

val_batch_gens = [torch.utils.data.DataLoader(test_datasets[i],
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=4) for i in range(DATASET_QUANTITY)]

# was weight_decay=3e-5
optimizers = [torch.optim.Adam(nn_modules[i].parameters(), lr=1e-3, weight_decay=3e-5) for i in range(DATASET_QUANTITY)]

NUM_EPOCHS = 20
train_loss = [list() for _ in range(DATASET_QUANTITY)]
results = [list() for _ in range(DATASET_QUANTITY)]
for epoch in range(NUM_EPOCHS):
    print("Epoch", len(results[0]))
    for i in range(DATASET_QUANTITY):
        nn_modules[i].train(True)
        optimizers[i].zero_grad()

    for idx, batch in tqdm.tqdm(enumerate(zip(*train_batch_gens))):
        for i in range(DATASET_QUANTITY):
            if True:
                for p in nn_modules[i].encoder.parameters():
                    p.requires_grad = False
            else:
                for p in nn_modules[i].encoder.parameters():
                    p.requires_grad = True

            if i < DATASET_QUANTITY * 3 // 4:
                pos_images, neg_images, _, _ = batch[i]
                pos_images = pos_images.to(device)
                neg_images = neg_images.to(device)

                pos_predictions = nn_modules[i](pos_images)
                neg_predictions = nn_modules[i](neg_images)
                loss = nn_modules[i].compute_pairwise_loss(
                    pos_predictions,
                    neg_predictions,
                )
                train_loss[i].append(loss.cpu().data.numpy())
            else:
                X_batch = batch[i][0].to(device)
                y_batch = batch[i][1].to(device)
                predictions = nn_modules[i](X_batch)
                loss = nn_modules[i].compute_loss(predictions,y_batch)
                train_loss[i].append(loss.cpu().data.numpy())

            loss.backward()
            if (idx + 1) % 10 == 0 or idx == len(train_batch_gens[i]) - 1:
                optimizers[i].step()
                optimizers[i].zero_grad()
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
            for batch_idx, batch in enumerate(val_batch_gens[i]):
                if i < DATASET_QUANTITY * 3 // 4:
                    pos_images, neg_images, _, _ = batch
                    pos_images = pos_images.to(device)
                    neg_images = neg_images.to(device)
                    pos_predictions = nn_modules[i](pos_images)
                    neg_predictions = nn_modules[i](neg_images)
                    loss = nn_modules[i].compute_pairwise_loss(
                        pos_predictions,
                        neg_predictions,
                    )
                    train_loss[i].append(loss.cpu().data.numpy())
                else:
                    X_batch, y_batch = batch[0], batch[1]
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
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
        metric = list()

        with torch.no_grad():
            for batch_idx, batch in enumerate(train_batch_gens[i]):
                if i < DATASET_QUANTITY * 3 // 4:
                    pos_images, neg_images, pos_labels, neg_labels = batch
                    pos_images = pos_images.to(device)
                    neg_images = neg_images.to(device)
                    X_batch = torch.cat([pos_images, neg_images])
                    y_batch = torch.cat([pos_labels, neg_labels])
                    y_pred = nn_modules[i](X_batch)
                    y_pred = nn_modules[i].post_processing(y_pred)
                    metric.append(nn_modules[i].metric(y_batch, y_pred.cpu()))
                else:
                    X_batch, y_batch = batch[0], batch[1]
                    X_batch = X_batch.to(device)
                    y_pred = nn_modules[i](X_batch)
                    y_pred = nn_modules[i].post_processing(y_pred)
                    metric.append(nn_modules[i].metric(y_batch, y_pred.cpu()))
            # for X_batch, y_batch in train_batch_gens[i]:
            #     X_batch = X_batch.to(device)
            #     y_pred = nn_modules[i](X_batch)
            #     y_pred = nn_modules[i].post_processing(y_pred)
            #     metric.append(nn_modules[i].metric(y_batch, y_pred.cpu()))
        print(i, np.mean(metric))
        results[i][-1].append(np.mean(metric))


    print("validation metric")
    for i in range(DATASET_QUANTITY):
        nn_modules[i].train(False)
        metric = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_batch_gens[i]):
                if i < DATASET_QUANTITY * 3 // 4:
                    pos_images, neg_images, pos_labels, neg_labels = batch
                    pos_images = pos_images.to(device)
                    neg_images = neg_images.to(device)
                    X_batch = torch.cat([pos_images, neg_images])
                    y_batch = torch.cat([pos_labels, neg_labels])
                    y_pred = nn_modules[i](X_batch)
                    y_pred = nn_modules[i].post_processing(y_pred)
                    metric.append(nn_modules[i].metric(y_batch, y_pred.cpu()))
                else:
                    X_batch, y_batch = batch[0], batch[1]
                    X_batch = X_batch.to(device)
                    y_pred = nn_modules[i](X_batch)
                    y_pred = nn_modules[i].post_processing(y_pred)
                    metric.append(nn_modules[i].metric(y_batch, y_pred.cpu()))
            # for X_batch, y_batch in val_batch_gens[i]:
            #     X_batch = X_batch.to(device)
            #     y_pred = nn_modules[i](X_batch)
            #     y_pred = nn_modules[i].post_processing(y_pred)
            #     metric.append(nn_modules[i].metric(y_batch, y_pred.cpu()))
        print(i, np.mean(metric))
        results[i][-1].append(np.mean(metric))

    # with open('results_train_' + str(DATASET_QUANTITY) + '.pkl', 'wb') as f:
    #     pickle.dump(results, f)

    # for i in range(DATASET_QUANTITY):
    #     torch.save(nn_modules[i].state_dict(), "nn_module_train_" + str(DATASET_QUANTITY) + '_' + str(i) + ".pt")
    #     torch.save(optimizers[i].state_dict(), "optimizer_" + str(DATASET_QUANTITY) + '_' + str(i) + ".pt")

