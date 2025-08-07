# torch
import torch
from torch.utils import data
# 3rd party
import cv2
# self
from src.utils import *

def img_croping(input1_img,input2_img, label_img, target_size):
    w, h = input1_img.shape[0], input1_img.shape[1]
    w_t, h_t = target_size[0], target_size[1]
    i = np.random.randint(0, w-w_t+1)
    j = np.random.randint(0, h-h_t+1)
    input1_img = input1_img[i:i+w_t, j:j+h_t]
    input2_img = input2_img[i:i+w_t, j:j+h_t]
    label_img = label_img[i:i+w_t, j:j+h_t]
    return input1_img,input2_img, label_img

def img_croping_finetune(input1_img,input2_img, label_img, target_size):
    w, h = input1_img.shape[0], input1_img.shape[1]
    w_t, h_t = target_size[0], target_size[1]
    i = np.random.randint(0, w-w_t+1)
    j = np.random.randint(0, h-h_t+1)
    input1_img = input1_img[i:i+w_t, j:j+h_t]
    input2_img = input2_img[i:i+w_t, j:j+h_t]
    w, h = label_img.shape[0], label_img.shape[1]
    i = np.random.randint(0, w-w_t+1)
    j = np.random.randint(0, h-h_t+1)
    label_img = label_img[i:i+w_t, j:j+h_t]
    return input1_img,input2_img, label_img

def img_croping_for_64(input_img):
    w, h = input_img.shape[0], input_img.shape[1]
    w_gap = divmod(w,64)
    h_gap = divmod(h,64)
    input_img = input_img[int(w_gap[1]/2):w-(w_gap[1]-int(w_gap[1]/2)), int(h_gap[1]/2):h-(h_gap[1]-int(h_gap[1]/2))]
    return input_img

class GenerateDataset(data.Dataset):
    def __init__(self, data_info, args=None):
        self.data_info = data_info
        self.args = args
    
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        source1_img_path, source2_img_path, target_img_path = self.data_info[idx]
        source1_img = cv2.imread(source1_img_path)
        source2_img = cv2.imread(source2_img_path)
        target_img = cv2.imread(target_img_path)

        if self.args.aug_list is not None:
            source1_img,source2_img, target_img = image_augment(source1_img,source2_img, target_img, self.args.aug_list)
        if self.args.target_size is not None:
            source1_img,source2_img, target_img = img_croping(source1_img,source2_img, target_img, self.args.target_size)
        else:
            source1_img = img_croping_for_64(source1_img)
            source2_img = img_croping_for_64(source2_img)
            target_img = img_croping_for_64(target_img)

        source1_img = source1_img[:, :, ::-1].copy()
        source2_img = source2_img[:, :, ::-1].copy()
        target_img = target_img[:, :, ::-1].copy()

        source1_img = torch.from_numpy(source1_img.transpose((2, 0, 1))).float().div(255.0)
        source2_img = torch.from_numpy(source2_img.transpose((2, 0, 1))).float().div(255.0)
        target_img = torch.from_numpy(target_img.transpose((2, 0, 1))).float().div(255.0)

        return source1_img, source2_img, target_img


class GenerateDataset_without_target(data.Dataset):
    def __init__(self, data_info, args=None):
        self.data_info = data_info
        self.args = args
    
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        source1_img_path,source2_img_path = self.data_info[idx]
        source1_img = cv2.imread(source1_img_path)
        source2_img = cv2.imread(source2_img_path)
        source1_img = img_croping_for_64(source1_img)
        source2_img = img_croping_for_64(source2_img)

        source1_img = source1_img[:, :, ::-1].copy()
        source2_img = source2_img[:, :, ::-1].copy()
        source1_img = torch.from_numpy(source1_img.transpose((2, 0, 1))).float().div(255.0)
        source2_img = torch.from_numpy(source2_img.transpose((2, 0, 1))).float().div(255.0)

        return source1_img, source2_img