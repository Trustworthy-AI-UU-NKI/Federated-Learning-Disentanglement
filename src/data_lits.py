import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
from sklearn.model_selection import train_test_split, KFold
from utils import Config
import flwr as fl
import os
import shutil
import configparser
import pickle
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import nibabel as nib
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

DATA_ORIGINAL_ROOT = "../lits_data_original/"
DATA_SOURCE_ROOT = "data_lits/"
DATA_CENTER_SPLITS_ROOT = DATA_SOURCE_ROOT + "splits/center_splits/"

LITS_CENTERS = [1, 2, 3, 4, 5]

def load_lits_centralized(center_nr, data_config):
    train_data = LiTS(center_nr, data_config, "train")
    val_data = LiTS(center_nr, data_config, "val")
    in_test_data = LiTS(center_nr, data_config, "in_test")

    if data_config["out_center"] is not None and (center_nr is None or center_nr == data_config["out_center"]):
        out_test_data = LiTS(data_config["out_center"], data_config, "out_test")
    else:
        out_test_data = []

    train_loader = DataLoader(train_data, data_config["batch_size"], shuffle=True, drop_last=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8)
    in_test_loader = DataLoader(in_test_data, batch_size=1, shuffle=False, num_workers=8)
    out_test_loader = DataLoader(out_test_data, batch_size=1, shuffle=False, num_workers=8)

    num_examples = {'train': len(train_data), 'val': len(val_data),
                    'in_test': len(in_test_data), 'out_test': len(out_test_data)}

    return {'train': train_loader, 'val': val_loader,
            'in_test': in_test_loader, 'out_test':out_test_loader}, num_examples

def load_lits_federated(center_nr, data_config):
    train_data = LiTS(center_nr, data_config, "train") if center_nr is not None else []
    val_data = LiTS(center_nr, data_config, "val")
    in_test_data = LiTS(center_nr, data_config, "in_test") if center_nr is None else []

    if data_config["out_center"] is not None and center_nr is None:
        out_test_data = LiTS(data_config["out_center"], data_config, "out_test")
    else:
        out_test_data = []

    train_loader = DataLoader(train_data, data_config["batch_size"],
                              shuffle=True, drop_last=True,
                              num_workers=8) if len(train_data) > 0 else None
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8)
    in_test_loader = DataLoader(in_test_data, batch_size=1, shuffle=False, num_workers=8)
    out_test_loader = DataLoader(out_test_data, batch_size=1, shuffle=False, num_workers=8)

    num_examples = {'train': len(train_data), 'val': len(val_data),
                    'in_test': len(in_test_data), 'out_test': len(out_test_data)}

    return {'train': train_loader, 'val': val_loader,
            'in_test': in_test_loader, 'out_test':out_test_loader}, num_examples

class LiTS(Dataset):
    def __init__(self, center_nr, data_config, mode):
        self.center_nr = center_nr
        self.load_in_ram = data_config["load_in_ram"]
        self.load_names = data_config["load_names"]
        out_center = data_config['out_center']

        if self.center_nr is None:
            self.imgs, self.masks = self.load_images_all_centers(out_center, mode)
        else:
            self.imgs, self.masks = self.load_images_center(center_nr, mode)

        target_size = data_config["target_size"]

        if mode == "train":
            self.transform = A.Compose([
                A.Normalize(mean=(0), std=(1)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(),
                # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.3),
                ToTensorV2()
        ])
        else:
            self.transform =  A.Compose([
                A.Normalize(mean=(0), std=(1)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.load_in_ram:
            if self.load_names:
                org_img = self.imgs[idx][0]
                org_mask = self.masks[idx][0]
            else:
                org_img = self.imgs[idx]
                org_mask = self.masks[idx]
        else:
            org_img = io.imread(self.imgs[idx], as_gray=True)
            org_mask = io.imread(self.masks[idx], as_gray=True)

        org_mask = (np.where(org_mask > 128, 1, 0)).astype(np.int64)

        data = self.transform(image=org_img, mask=org_mask)
        img, mask = data['image'], data['mask'].type(torch.LongTensor).unsqueeze(dim=0)

        if self.load_in_ram:
            if self.load_names:
                return img, mask, self.imgs[idx][1], self.masks[idx][1]
            else:
                return img, mask
        else:
            return img, mask, self.imgs[idx], self.masks[idx]

    def load_image_mask_pairs(self, df, imgs, masks):
        for i in range(df.shape[0]):
            if self.load_in_ram:
                img = (io.imread(df['image'][i], as_gray=True), df['image'][i]) if self.load_names else io.imread(df['image'][i], as_gray=True)
                mask = (io.imread(df['mask'][i], as_gray=True), df['mask'][i]) if self.load_names else io.imread(df['mask'][i], as_gray=True)
            else:
                img = df['image'][i]
                mask = df['mask'][i]

            imgs.append(img)
            masks.append(mask)

    def load_images_center(self, center_nr, mode):
        imgs = []
        masks = []

        if mode == "out_test":
            modes = ["train", "val", "in_test"]
        else:
            modes = [mode]

        for mode in modes:
            path = os.path.join(DATA_CENTER_SPLITS_ROOT, f"c{center_nr}/{mode}_split.csv")
            df = pd.read_csv(path)
            self.load_image_mask_pairs(df, imgs, masks)

        return imgs, masks


    def load_images_all_centers(self, out_center, mode):
        imgs = []
        masks = []

        centers = LITS_CENTERS.copy()
        if out_center is not None:
            centers.remove(out_center)

        for center_nr in centers:
            imgs_c, masks_c = self.load_images_center(center_nr, mode)
            imgs.extend(imgs_c)
            masks.extend(masks_c)

        return imgs, masks
    
def create_center_splits_lits(train_ratio=0.8):
    if os.path.isdir(DATA_CENTER_SPLITS_ROOT):
        print(f"Deleting existing {DATA_CENTER_SPLITS_ROOT} directory...")
        shutil.rmtree(DATA_CENTER_SPLITS_ROOT)

    os.mkdir(DATA_CENTER_SPLITS_ROOT)

    for center_nr in LITS_CENTERS:
        img_names_per_volume = defaultdict(lambda: [])
        mask_names_per_volume = defaultdict(lambda: [])

        save_path = os.path.join(DATA_CENTER_SPLITS_ROOT, f"c{center_nr}")
        os.mkdir(save_path)

        img_path = os.path.join(DATA_SOURCE_ROOT, f"data_C{center_nr}/images_C{center_nr}")
        mask_path = os.path.join(DATA_SOURCE_ROOT, f"data_C{center_nr}/masks_C{center_nr}")

        for filename in sorted(os.listdir(img_path)):
            img_nr = int(filename.split('_')[0][5:])

            img_names_per_volume[img_nr].append(os.path.join(img_path, filename))
            mask_names_per_volume[img_nr].append(os.path.join(mask_path, f"mask{filename[5:].split('.')[0]}.jpg"))

        train_idx, rest_idx = train_test_split(list(img_names_per_volume.keys()),
                                                train_size=train_ratio,
                                                random_state=0)
        val_idx, test_idx = train_test_split(rest_idx,
                                            train_size=0.5,
                                            random_state=0)
        
        train_imgs = [name for i in train_idx for name in img_names_per_volume[i]]
        val_imgs = [name for i in val_idx for name in img_names_per_volume[i]]
        test_imgs = [name for i in test_idx for name in img_names_per_volume[i]]

        train_masks = [name for i in train_idx for name in mask_names_per_volume[i]]
        val_masks = [name for i in val_idx for name in mask_names_per_volume[i]]
        test_masks = [name for i in test_idx for name in mask_names_per_volume[i]]

        for split in [("train", train_imgs, train_masks),
                      ("val", val_imgs, val_masks),
                      ("in_test", test_imgs, test_masks)]:
            split_name, imgs, masks = split
            df = pd.DataFrame({'image': imgs, 'mask': masks})
            df.to_csv(os.path.join(save_path, f"{split_name}_split.csv"))

        print(f"Created train/val/test splits for center {center_nr}...")
    
def convert_nii_to_jpg():
    N_SLICES = 28 # based on lowest n_slices that contain liver out of all scans 

    for center_nr in LITS_CENTERS:
        center_path = DATA_SOURCE_ROOT + f"data_C{center_nr}/"
        if os.path.exists(center_path):
            shutil.rmtree(center_path)
        os.mkdir(center_path)
        os.mkdir(center_path + f"images_C{center_nr}")
        os.mkdir(center_path + f"masks_C{center_nr}")

    for filename in tqdm(sorted(os.listdir(DATA_ORIGINAL_ROOT + "images"))):
        img_path = os.path.join(DATA_ORIGINAL_ROOT + "images", filename)
        nr = filename.split('-')[-1].split('.')[0]
        center_nr = get_center_nr(nr)

        mask_path = os.path.join(DATA_ORIGINAL_ROOT + "masks", f"segmentation-{nr}.nii")
        img_org = nib.load(img_path).get_fdata()
        mask_org = nib.load(mask_path).get_fdata()

        img_clipped = img_org
        img_clipped[img_clipped < -200] = -200
        img_clipped[img_clipped > 200] = 200

        img_min = np.min(img_clipped)
        img_max = np.max(img_clipped)
        img_clipped = ((img_clipped - img_min) / (img_max - img_min)) * 255
        img = img_clipped.astype(np.uint8)

        min_slice, max_slice = get_liver_slice_range(mask_org)

        # slice_indices = np.linspace(min_slice, max_slice, N_SLICES)
        # slice_indices = [round(val) for val in slice_indices]
        slice_indices = range(min_slice, max_slice+1)

        for slice_i in slice_indices:
            slice_mask_org = mask_org[:, :, slice_i]

            new_img_filename = f"image{nr}_slice{slice_i}.jpg"
            new_mask_filename = f"mask{nr}_slice{slice_i}.jpg"
            slice_img = img[:, :, slice_i]

            # set liver segmentation to 0 and tumor to 255
            slice_mask_org[slice_mask_org < 2] = 0
            slice_mask_org[slice_mask_org == 2] = 255
            slice_mask = slice_mask_org.astype(np.uint8)

            if np.sum(slice_mask / 255) > 50:
                io.imsave(DATA_SOURCE_ROOT + f"data_C{center_nr}/images_C{center_nr}/" + new_img_filename, slice_img)
                io.imsave(DATA_SOURCE_ROOT + f"data_C{center_nr}/masks_C{center_nr}/" + new_mask_filename, slice_mask)

def get_liver_slice_range(mask):
    min_slice, max_slice = None, None

    for i in range(mask.shape[-1]):
        slice_i = mask[:, :, i]
        if np.sum(mask[:, :, i]) > 0:
            min_slice = i
            break

    for i in range(mask.shape[-1]-1, -1, -1):
        slice_i = mask[:, :, i]
        if np.sum(mask[:, :, i]) > 0:
            max_slice = i
            break

    return min_slice, max_slice

def get_center_nr(file_nr):
     file_nr = int(file_nr)
     if file_nr < 30:
          return 1
     elif file_nr < 60:
          return 2
     elif file_nr < 90:
          return 3
     elif file_nr < 120:
          return 4
     else:
          return 5
     
def create_lits_data_details(center_nr, small_threshold=1000):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 1
    params.maxArea = 512*512

    params.filterByColor = True
    params.blobColor = 255

    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByCircularity = False

    detector = cv2.SimpleBlobDetector_create(params)

    mask_path = f"data_lits/data_C{center_nr}/masks_C{center_nr}"
    new_path = f"data_lits/data_C{center_nr}/images_C{center_nr}"
    save_path = f"data_lits/metadata/my_dataDetails_C{center_nr}.csv"

    df = pd.DataFrame(columns=['img_name', 'small_polyp', 'inv_area'])

    for i, filename in enumerate(sorted(os.listdir(mask_path))):
        small_polyp = False
        full_path = os.path.join(new_path, filename.replace('mask', 'image'))

        im = cv2.imread(os.path.join(mask_path, filename), cv2.IMREAD_GRAYSCALE)
        # im = cv2.resize(im, (512, 512))

        # empty mask
        if np.sum(im/255) == 0:
            inv_area = 0
            df.loc[i] = [full_path, False, 0]
            continue

        inv_area = (512*512) / np.sum(im/255)

        if inv_area >= small_threshold:
            small_polyp = True

        df.loc[i] = [full_path, small_polyp, inv_area]

    df.to_csv(save_path, index=False)

def estimate_area(im_org, keypoints):
    sorted_keypoints = sorted(keypoints, key=lambda x: x.size)
    keypoint = sorted_keypoints[0]
    x, y = keypoint.pt
    x_c, y_c = round(x), round(y)

    x_min, x_max, y_min, y_max = None, None, None, None
    x_l, x_r, y_t, y_b = 0, 0, 0, 0

    # Get (rough) bounding box coordinates
    while x_min is None or x_max is None or y_min is None or y_max is None:
        if x_c - x_l >= 0 and im_org[y_c][x_c - x_l] == 255:
            x_l += 1
        else:
            x_min = x_c - x_l + 1

        if x_c + x_r < 512 and im_org[y_c][x_c + x_r] == 255:
            x_r += 1
        else:
            x_max = x_c + x_r - 1

        if y_c - y_t >= 0 and im_org[y_c - y_t][x_c] == 255:
            y_t += 1
        else:
            y_min = y_c - y_t + 1

        if y_c + y_b < 512 and im_org[y_c + y_b][x_c] == 255:
            y_b += 1
        else:
            y_max = y_c + y_b - 1

    estimated_area = (x_max - x_min) * (y_max - y_min)

    if estimated_area == 0: # set estimated_area to 1 pixel if zero
        return 1
    
    return estimated_area

if __name__ == "__main__":
    # convert_nii_to_jpg()
    # create_center_splits_lits()

    for center_nr in range(1,6):
        create_lits_data_details(center_nr)
