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

import nibabel as nib

DATA_SOURCE_ROOT = "data_polypgen/"
DATA_CENTER_SPLITS_ROOT = DATA_SOURCE_ROOT + "splits/center_splits/"
DATA_PATIENT_SPLITS_ROOT = DATA_SOURCE_ROOT + "splits/patient_splits/"
DATA_FOLD_SPLITS_ROOT = DATA_SOURCE_ROOT + "splits/fold_splits/"
CKPTS_ROOT = "ckpts/"
CFGS_ROOT = "cfgs/"
RESULTS_ROOT = "results/"
CENTERS_TXTFILE = "temp_centers_textfile_"

POLYP_CENTERS = [1, 2, 3, 4, 5, 6]
LITS_CENTERS = [1, 2, 3, 4, 5]

def load_model_parameters(ckpt_name):
    state_dict = torch.load(os.path.join(CKPTS_ROOT, f"{ckpt_name}.pt"))
    params = [val.cpu().numpy() for _, val in state_dict.items()]
    return fl.common.ndarrays_to_parameters(params)

def load_polypgen_centralized(center_nr, data_config):
    train_data = PolypGen(center_nr, data_config, "train")
    val_data = PolypGen(center_nr, data_config, "val")
    in_test_data = PolypGen(center_nr, data_config, "in_test")

    if data_config["out_center"] is not None and (center_nr is None or center_nr == data_config["out_center"]):
        out_test_data = PolypGen(data_config["out_center"], data_config, "out_test")
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

def load_polypgen_federated(center_nr, data_config):
    train_data = PolypGen(center_nr, data_config, "train") if center_nr is not None else []
    val_data = PolypGen(center_nr, data_config, "val")
    in_test_data = PolypGen(center_nr, data_config, "in_test") if center_nr is None else []

    if data_config["out_center"] is not None and center_nr is None:
        out_test_data = PolypGen(data_config["out_center"], data_config, "out_test")
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

def load_config(cfg_name="default"):
    config_p = configparser.ConfigParser()
    config_p.read(os.path.join(CFGS_ROOT, f"{cfg_name}.cfg"))
    return Config(config_p, cfg_name)

def load_model_checkpoint(net, config_name, suffix="best"):
    ckpt = torch.load(os.path.join(CKPTS_ROOT, f"{config_name}_{suffix}.pt"))
    net.load_state_dict(ckpt)

def save_model(net, config_name, last=False):
    suffix = "last" if last else "best"
    torch.save(net.state_dict(), os.path.join(CKPTS_ROOT, f"{config_name}_{suffix}.pt"))

def save_results(results, config, filename=None):
    if filename is None:
        filename = f"{config.server.strategy}_" + \
                   f"{config.client.model}_rounds" + \
                   f"{config.server.num_rounds}_ep" + \
                   f"{config.client.epochs}_lr" + \
                   f"{config.client.lr}_outcenter" + \
                   f"{config.data.out_center}"
    with open(os.path.join(RESULTS_ROOT, f"{filename}.pkl"), 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_results(filename):
    with open(os.path.join(RESULTS_ROOT, f"{filename}.pkl"), 'rb') as f:
        results = pickle.load(f)
    return results

class PolypGen(Dataset):
    def __init__(self, center_nr, data_config, mode):
        self.center_nr = center_nr
        self.load_in_ram = data_config["load_in_ram"]
        self.load_names = data_config["load_names"]
        out_center, splits = data_config['out_center'], data_config['splits']
        fold_nr = data_config['fold_nr']

        if splits == "per_center":
            if self.center_nr is None:
                self.imgs, self.masks = self.load_images_all_centers(out_center, mode)
            else:
                self.imgs, self.masks = self.load_images_center(center_nr, mode)
        elif splits == "per_patient":
            if self.center_nr is None:
                self.imgs, self.masks = self.load_images_all_patients(out_center, mode)
            elif self.center_nr == out_center and mode == "out_test":
                self.imgs, self.masks = self.load_images_center(center_nr, mode)
            else:
                self.imgs, self.masks = self.load_images_patients(center_nr, out_center, mode)
        elif splits == "per_fold":
            if self.center_nr is None:
                self.imgs, self.masks = self.load_images_all_fold(out_center, fold_nr, mode)
            elif self.center_nr == out_center and mode == "out_test":
                self.imgs, self.masks = self.load_images_center(center_nr, mode)
            else:
                self.imgs, self.masks = self.load_images_fold(center_nr, fold_nr, mode)
        else:
            print(f"Splits type {splits} is not supported!")

        target_size = data_config["target_size"]

        if mode == "train":
            self.transform = A.Compose([
                A.Resize(target_size, target_size),
                A.Normalize(),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.3),
                ToTensorV2()
        ])
        else:
            self.transform =  A.Compose([
                A.Resize(target_size, target_size),
                A.Normalize(),
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
            org_img = io.imread(self.imgs[idx])
            org_mask = io.imread(self.masks[idx])

        org_mask = (np.where(org_mask > 128, 1, 0)).astype(np.uint64)

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
                img = (io.imread(df['image'][i]), df['image'][i]) if self.load_names else io.imread(df['image'][i])
                mask = (io.imread(df['mask'][i]), df['mask'][i]) if self.load_names else io.imread(df['mask'][i])
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

    def load_images_patients(self, center_nr, out_center, mode):
        imgs = []
        masks = []

        base_name = f"no_centre_{out_center}" if out_center is not None else "all_centres"
        path = os.path.join(DATA_PATIENT_SPLITS_ROOT, f"{base_name}_{mode}_split.csv")
        df = pd.read_csv(path)
        start_ix = len(DATA_SOURCE_ROOT)
        data_c = f"data_C{center_nr}"

        for i in range(df.shape[0]):
            if df['image'][i][start_ix:start_ix+7] == data_c:
                if self.load_in_ram:
                    img = (io.imread(df['image'][i]), df['image'][i]) if self.load_names else io.imread(df['image'][i])
                    mask = (io.imread(df['mask'][i]), df['mask'][i]) if self.load_names else io.imread(df['mask'][i])
                else:
                    img = df['image'][i]
                    mask = df['mask'][i]

                imgs.append(img)
                masks.append(mask)

        return imgs, masks
    
    def load_images_fold(self, center_nr, fold_nr, mode):
        imgs = []
        masks = []

        if mode == "out_test":
            modes = ["train", "val", "in_test"]
        else:
            modes = [mode]

        for mode in modes:
            path = os.path.join(DATA_FOLD_SPLITS_ROOT, f"c{center_nr}/fold{fold_nr}/{mode}_split.csv")
            df = pd.read_csv(path)
            self.load_image_mask_pairs(df, imgs, masks)

        return imgs, masks

    def load_images_all_centers(self, out_center, mode):
        imgs = []
        masks = []

        centers = POLYP_CENTERS.copy()
        if out_center is not None:
            centers.remove(out_center)

        for center_nr in centers:
            imgs_c, masks_c = self.load_images_center(center_nr, mode)
            imgs.extend(imgs_c)
            masks.extend(masks_c)

        return imgs, masks

    def load_images_all_patients(self, out_center, mode):
        imgs = []
        masks = []

        base_name = f"no_centre_{out_center}" if out_center is not None else "all_centres"
        path = os.path.join(DATA_PATIENT_SPLITS_ROOT, f"{base_name}_{mode}_split.csv")
        df = pd.read_csv(path)
        self.load_image_mask_pairs(df, imgs, masks)

        return imgs, masks
    
    def load_images_all_fold(self, out_center, fold_nr, mode):
        imgs = []
        masks = []

        centers = POLYP_CENTERS.copy()
        if out_center is not None:
            centers.remove(out_center)

        for center_nr in centers:
            imgs_c, masks_c = self.load_images_fold(center_nr, fold_nr, mode)
            imgs.extend(imgs_c)
            masks.extend(masks_c)

        return imgs, masks


def create_center_splits_polypgen(train_ratio=0.8):
    if os.path.isdir(DATA_CENTER_SPLITS_ROOT):
        print(f"Deleting existing {DATA_CENTER_SPLITS_ROOT} directory...")
        shutil.rmtree(DATA_CENTER_SPLITS_ROOT)

    os.mkdir(DATA_CENTER_SPLITS_ROOT)

    for center_nr in POLYP_CENTERS:
        img_names = []
        mask_names = []

        save_path = os.path.join(DATA_CENTER_SPLITS_ROOT, f"c{center_nr}")
        os.mkdir(save_path)

        img_path = os.path.join(DATA_SOURCE_ROOT, f"data_C{center_nr}/images_C{center_nr}")
        mask_path = os.path.join(DATA_SOURCE_ROOT, f"data_C{center_nr}/masks_C{center_nr}")

        for filename in sorted(os.listdir(img_path)):
            img_names.append(os.path.join(img_path, filename))
            mask_names.append(os.path.join(mask_path, f"{filename[:-4]}_mask.jpg"))

        train_imgs, rest_imgs, train_masks, rest_masks = train_test_split(img_names,
                                                                          mask_names,
                                                                          train_size=train_ratio,
                                                                          random_state=0)
        val_imgs, test_imgs, val_masks, test_masks = train_test_split(rest_imgs,
                                                                       rest_masks,
                                                                       train_size=0.5,
                                                                       random_state=0)

        for split in [("train", train_imgs, train_masks),
                      ("val", val_imgs, val_masks),
                      ("in_test", test_imgs, test_masks)]:
            split_name, imgs, masks = split
            df = pd.DataFrame({'image': imgs, 'mask': masks})
            df.to_csv(os.path.join(save_path, f"{split_name}_split.csv"))

        print(f"Created train/val/test splits for center {center_nr}...")

def create_fold_splits_polypgen(k=5):
    if os.path.isdir(DATA_FOLD_SPLITS_ROOT):
        print(f"Deleting existing {DATA_FOLD_SPLITS_ROOT} directory...")
        shutil.rmtree(DATA_FOLD_SPLITS_ROOT)

    os.mkdir(DATA_FOLD_SPLITS_ROOT)

    for center_nr in POLYP_CENTERS:
        img_names = []
        mask_names = []

        save_path = os.path.join(DATA_FOLD_SPLITS_ROOT, f"c{center_nr}")
        os.mkdir(save_path)

        img_path = os.path.join(DATA_SOURCE_ROOT, f"data_C{center_nr}/images_C{center_nr}")
        mask_path = os.path.join(DATA_SOURCE_ROOT, f"data_C{center_nr}/masks_C{center_nr}")

        for filename in sorted(os.listdir(img_path)):
            img_names.append(os.path.join(img_path, filename))
            mask_names.append(os.path.join(mask_path, f"{filename[:-4]}_mask.jpg"))

        img_names, mask_names = np.array(img_names), np.array(mask_names)
        kf = KFold(n_splits=k)
        for i, fold_idx in enumerate(kf.split(img_names)):
            fold_path = os.path.join(save_path, f"fold{i+1}")
            os.mkdir(fold_path)
            train_idx, test_idx = fold_idx

            train_imgs, rest_imgs, train_masks, rest_masks = img_names[train_idx], img_names[test_idx], \
                                                             mask_names[train_idx], mask_names[test_idx]

            val_imgs, test_imgs, val_masks, test_masks = train_test_split(rest_imgs,
                                                                        rest_masks,
                                                                        train_size=0.5,
                                                                        random_state=0)
            
            for split in [("train", train_imgs, train_masks),
                        ("val", val_imgs, val_masks),
                        ("in_test", test_imgs, test_masks)]:
                split_name, imgs, masks = split
                df = pd.DataFrame({'image': imgs, 'mask': masks})
                df.to_csv(os.path.join(fold_path, f"{split_name}_split.csv"))

        print(f"Created {k} folds with train/val/test splits for center {center_nr}...")

def create_centers_textfile(config, out_center=6):
    exp_name = config.name
    centers = POLYP_CENTERS.copy() if config.data.dataset == "polypgen" else LITS_CENTERS.copy()
    centers.remove(out_center)

    filename = CENTERS_TXTFILE + exp_name + ".txt"

    if os.path.exists(filename):
        os.remove(filename)

    centers_str = ' '.join([str(center) for center in centers])

    with open(filename, 'w') as f:
        f.write(centers_str)

def delete_centers_textfile(exp_name):
    filename = CENTERS_TXTFILE + exp_name + ".txt"

    if os.path.exists(filename):
        os.remove(filename)

def get_center_from_textfile(exp_name):
    filename = CENTERS_TXTFILE + exp_name + ".txt"

    with open(filename, 'r+') as f:
        centers_str = f.read()
        centers = centers_str.split(' ')

        if centers[0] == '':
            raise ValueError("Center list in textfile is empty! It needs to be recreated again before calling this function.")

        # remove old centers list
        f.seek(0)
        f.truncate()

        new_centers_str = ' '.join(centers[1:])
        f.write(new_centers_str)

    return int(centers[0])

def check_small_polyp(img_path, my_details, dataset):
    center_nr = img_path.split('images_C')[-1][0]
    prefix = 'my_' if my_details else ''
    data_path = "data" if dataset == "polypgen" else "data_lits" 
    meta_path = f"{data_path}/metadata/{prefix}dataDetails_C{center_nr}.csv"
    df = pd.read_csv(meta_path)

    if not my_details:
        row = df.loc[df['fileList'] == img_path].iloc[0]
        size_list = [int(val) for val in row['sizeList'].strip('[]').split(', ')]
        small_polyp = True if size_list[1] > 0 else False
        inv_area = None
    else:
        row = df.loc[df['img_name'] == img_path].iloc[0]
        small_polyp = row['small_polyp']
        inv_area = row['inv_area']

    return small_polyp, inv_area

def get_polyp_size(img_path):
    center_nr = img_path.split('images_C')[-1][0]
    meta_path = f"data/metadata/dataDetails_C{center_nr}.csv"
    df = pd.read_csv(meta_path)
    row_val = df.loc[df['fileList'] == img_path]['sizeType'].values[0]
    size_list = [int(val) for val in row_val.strip('[]').split(', ')]

    if size_list[1] > 0:
        return "small"
    elif size_list[2] > 0:
        return "medium"
    elif size_list[3] > 0:
        return "large"
    else:
        return None

def remove_path_prefix():
    path = "data/metadata"

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        df = pd.read_csv(file_path)

        for index, row in df.iterrows():
            name = row['fileList']
            name = '/'.join(name.split('/')[7:])
            name = "data/" + name
            df.loc[index, 'fileList'] = name
        os.remove(file_path)
        df.to_csv(file_path, index=False)

def find_mismatched_files():
    meta_path = "data/metadata"
    img_path = "data"

    for center in range(1, 6):
        center_name = f"C{center}"
        meta_center_not_found = set()

        meta_center_path = os.path.join(meta_path, f"dataDetails_{center_name}.csv")
        img_center_path = os.path.join(img_path, f"data_{center_name}/images_{center_name}")

        meta_center_names = set(pd.read_csv(meta_center_path)['fileList'])
        img_center_names = set(os.listdir(img_center_path))

        for meta_center_name in meta_center_names:
            meta_name = meta_center_name.split('/')[-1]

            if meta_name not in img_center_names:
                meta_center_not_found.add(meta_center_name)
            else:
                img_center_names.remove(meta_name)

        print(f"Mismatched names for center: {center}")
        print("metadata:", meta_center_not_found)
        print("images:", img_center_names)

def create_data_details(center_nr, small_threshold=150):
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

    mask_path = f"data/data_C{center_nr}/masks_C{center_nr}"
    new_path = f"data/data_C{center_nr}/images_C{center_nr}"
    save_path = f"data/metadata/my_dataDetails_C{center_nr}.csv"

    df = pd.DataFrame(columns=['img_name', 'small_polyp', 'inv_area'])

    for i, filename in enumerate(sorted(os.listdir(mask_path))):
        small_polyp = False
        full_path = os.path.join(new_path, filename.split('_mask')[0] + ".jpg")

        im = cv2.imread(os.path.join(mask_path, filename), cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (512, 512))

        # empty mask
        if np.sum(im/255) == 0:
            inv_area = 0
            df.loc[i] = [full_path, False, 0]
            continue

        inv_area = (512*512) / np.sum(im/255)

        if inv_area < 20:
            kernel = np.ones((15, 15), np.uint8)
        else:
            kernel = np.ones((11, 11), np.uint8)

        im_org = im
        keypoints = detector.detect(im_org)

        if len(keypoints) == 0:
            small_polyp = True
        else:
            estimated_area = estimate_area(im_org, keypoints)
            # 200 based on 1920x1080 / 100x100 from PolypGen paper
            small_polyp = True if (512*512)/estimated_area >= small_threshold else False

            # Try again with keypoints from eroded image
            if not small_polyp:
                im_erode = cv2.erode(im, kernel, borderType=cv2.BORDER_REFLECT)
                keypoints = detector.detect(im_erode)
                estimated_area = estimate_area(im_org, keypoints) # use im_org, otherwise estimated area is too small
                # 200 based on 1920x1080 / 100x100 from PolypGen paper
                small_polyp = True if (512*512)/estimated_area >= small_threshold else False

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

    return (x_max - x_min) * (y_max - y_min)

def compare_data_details(center_nr):
    path1 = f"data/metadata/dataDetails_C{center_nr}.csv"
    path2 = f"data/metadata/my_dataDetails_C{center_nr}.csv"

    tp, tn, fp, fn = 0, 0, 0, 0

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    for i, row in df1.iterrows():
        name = row['fileList']
        size_type = [int(val) for val in row['sizeType'].strip('[]').split(', ')]

        if len(df2.loc[df2['img_name'] == name]) != 1:
            raise Exception("Image name should match one-to-one!")
        
        df2_row = df2.loc[df2['img_name'] == name].iloc[0]
        df2_small_polyp = df2_row['small_polyp']

        if df2_small_polyp and size_type[1] > 0:
            tp += 1
            print("correct", name, size_type, df2_small_polyp)
        elif not df2_small_polyp and size_type[1] == 0:   
            tn += 1
            print("correct", name, size_type, df2_small_polyp)
        elif df2_small_polyp and size_type[1] == 0:
            fp += 1
            print("incorrect", name, size_type, df2_small_polyp, df2_row['inv_area'])
        else:
            fn += 1
            print("incorrect", name, size_type, df2_small_polyp, df2_row['inv_area'])

    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Accuracy: {(tp+tn) / (tp+tn+fp+fn)}")
    print(f"Recall: {tp / (tp + fn)}")

if __name__ == "__main__":
    print("Creating train/val/test splits for all PolypGen centers...")
    create_center_splits_polypgen()