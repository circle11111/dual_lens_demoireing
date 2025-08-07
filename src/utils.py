# python
import os
import random
import datetime
# 3rd party
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import lpips
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# torch
import torch


# 设置随机种子
GLOBAL_SEED = 1999
GLOBAL_WORKER_ID = None
def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
def worker_init_fn(worder_id):
	global GLOBAL_WORKER_ID
	GLOBAL_WORKER_ID = worder_id
	np.random.seed(GLOBAL_WORKER_ID + worder_id)

# 读取图片路径信息
def get_img_info(data_path):
	data_info = list()
	img_names = os.listdir(data_path)
	img_names.sort()
	for i in range(len(img_names)):
		img_name = img_names[i]
		path_img = os.path.join(data_path, img_name)
		path_img = Path(path_img).as_posix()
		data_info.append(path_img)
	return data_info
def get_img_info_gan_gt(data_path, length):
	data_info = list()
	img_names = os.listdir(data_path)
	img_names.sort()
	state = np.random.get_state()
	np.random.shuffle(img_names)
	for i in range(length):
		img_name = img_names[i]
		path_img = os.path.join(data_path, img_name)
		path_img = Path(path_img).as_posix()
		data_info.append(path_img)
	return data_info
def get_img_info_dual(data_path):
	data_info = list()
	img_names = os.listdir(data_path)
	img_names.sort()
	for i in range(0, len(img_names), 2):
		img_name = img_names[i]
		path_img = os.path.join(data_path, img_name)
		path_img = Path(path_img).as_posix()
		data_info.append(path_img)
	return data_info
def get_img_info_dual_realcamera(data_path):
	data_info = list()
	img_names = os.listdir(data_path)
	img_names.sort()
	for i in range(0, len(img_names), 4):
		img_name = img_names[i]
		path_img = os.path.join(data_path, img_name)
		path_img = Path(path_img).as_posix()
		data_info.append(path_img)
	return data_info
	
def Shuffle(source_dataset, taregt_dataset, val_ratio):
	state = np.random.get_state()
	np.random.shuffle(source_dataset)
	np.random.set_state(state)
	np.random.shuffle(taregt_dataset)

	source_train_info = source_dataset[0:int(val_ratio * len(source_dataset))]
	source_val_info = source_dataset[int(val_ratio * len(source_dataset)):]

	target_train_info = taregt_dataset[0:int(val_ratio * len(taregt_dataset))]
	target_val_info = taregt_dataset[int(val_ratio * len(taregt_dataset)):]
	return source_train_info, source_val_info, target_train_info, target_val_info

def shuffle_dataset(source_dataset, target_dataset, train_data_path, val_data_path, val_ratio):
	source_train_info, source_val_info, target_train_info, target_val_info = Shuffle(source_dataset, target_dataset, val_ratio)
	train = np.column_stack((source_train_info, target_train_info))
	val = np.column_stack((source_val_info, target_val_info))
	pd.DataFrame(train, columns=["source", "target"]).to_csv(train_data_path, index=False, encoding='utf-8')
	pd.DataFrame(val, columns=["source", "target"]).to_csv(val_data_path, index=False, encoding='utf-8')

def get_shuffled_data(datapath):
	dataset = pd.read_csv(datapath, sep=',', encoding='utf-8')
	dataset = dataset.values.tolist()
	return dataset

def save_data_path(source1_dataset, source2_dataset, target_dataset, save_data_path):
	paths = np.column_stack((source1_dataset, source2_dataset, target_dataset))
	pd.DataFrame(paths, columns=["source1","source2", "target"]).to_csv(save_data_path, index=False, encoding='utf-8')

def get_path_data(datapath):
	dataset = pd.read_csv(datapath, sep=',', encoding='utf-8')
	dataset = dataset.values.tolist()
	return dataset

def image_augment(input1_img, input2_img, label_img, aug_list):
	random.shuffle(aug_list)
	for i, val in enumerate(aug_list):
		p = random.random()
		if val == "crop":
			# 0.8 的概率做裁剪
			if p > 0.2:
				input1_img,input2_img, label_img = crop_img(input1_img,input2_img, label_img)
		elif val == "level_flip":
			if p > 0.5:
				input1_img = level_flip(input1_img)
				input2_img = level_flip(input2_img)
				label_img = level_flip(label_img)
		elif val == "vertical_flip":
			if p > 0.8:
				input1_img = vertical_flip(input1_img)
				input2_img = vertical_flip(input2_img)
				label_img = vertical_flip(label_img)
		elif val == "rot":
			# 固定角度的旋转放一起, 随机概率看转 90 还是 180, 270
			if p > 0.5:
				angle_times = random.randint(1, 3)
				input1_img = np.rot90(input1_img, angle_times)
				input2_img = np.rot90(input2_img, angle_times)
				label_img = np.rot90(label_img, angle_times)
		elif val == "rotate":
			if p > 0.8:
				input1_img,input2_img, label_img = rotate_image(input1_img,input2_img, label_img)
				# label_img = rotate_image(label_img)
	return input1_img,input2_img, label_img

def crop_img(input1_img, input2_img, label_img):
	w, h = input1_img.shape[0], input1_img.shape[1]
	w_ratio, h_ratio = random.uniform(0.8, 0.95), random.uniform(0.8, 0.95)
	w_t, h_t = int(w*w_ratio), int(h*h_ratio)
	i = np.random.randint(0, w-w_t+1)
	j = np.random.randint(0, h-h_t+1)
	input1_img = input1_img[i:i+w_t, j:j+h_t]
	input2_img = input2_img[i:i+w_t, j:j+h_t]
	label_img = label_img[i:i+w_t, j:j+h_t]
	return input1_img,input2_img, label_img

def level_flip(img):
	img = cv2.flip(img, 1)
	return img

def vertical_flip(img):
	img = cv2.flip(img, 0)
	return img

def rotate_image(img1,img2, label):
	angle = np.random.randint(0, 30)
	h, w = img1.shape[0], img1.shape[1]
	ch, cw = h/2, w/2
	M = cv2.getRotationMatrix2D((ch, cw), angle, 1.0)
	img1 = cv2.warpAffine(img1, M, (h, w))
	img2 = cv2.warpAffine(img2, M, (h, w))
	label = cv2.warpAffine(label, M, (h, w))
	return img1,img2, label

def resize_image(img, target_size):
	img = cv2.resize(img, (target_size, target_size))
	return img

class PSNR_SSIM_LPIPS(object):
	def __init__(self, use_gpu=True):
		self.loss_fn = lpips.LPIPS(net='alex')
		if(use_gpu):
			self.loss_fn.cuda()
		self.Psnrs = []
		self.Ssims = []
		self.Lpips = []

	def get_info(self):
		return np.mean(self.Psnrs), np.mean(self.Ssims), np.mean(self.Lpips)

	def SSIM(self, target, ref):
		return compare_ssim(target, ref, channel_axis=2)

	def PSNR(self, img1, img2):
		return compare_psnr(img1, img2)

	def compute(self, preds, labels):
		list_psnr = []
		list_ssim = []
		list_lpips = []
		for p, l in zip(preds, labels):
			lpips_value = self.loss_fn(p, l)
			imgP = p.clone().cpu()
			# print(imgP.shape)
			imgP = torch.clamp(imgP, 0, 1).mul(255).byte().numpy().transpose((1, 2, 0))
			imgL = l.clone().cpu()
			imgL = torch.clamp(imgL, 0, 1).mul(255).byte().numpy().transpose((1, 2, 0))
			psnr = self.PSNR(imgP, imgL)
			ssim = self.SSIM(imgP, imgL)
			list_psnr.append(psnr)
			list_ssim.append(ssim)
			list_lpips.append(lpips_value.cpu().detach().numpy())
		self.Psnrs.append(np.mean(list_psnr))
		self.Ssims.append(np.mean(list_ssim))
		self.Lpips.append(np.mean(list_lpips))
		return (np.mean(list_psnr), np.mean(list_ssim), np.mean(list_lpips))

def log(*args, **kwargs):
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def alpha_bright_adjust(out, target, pixel_mask): #pixel_mask: nx3x256xHxW
	out_result = out - out
	# print(out.type)
	# print(target.type)
	# print(pixel_mask.type)
	i = 0
	for (out_i, target_i, pixel_mask_i) in zip(out, target, pixel_mask):
		out_result_i = out_i - out_i
		for c in range(3):
			out_result_i_c = out_i[c, :, :]-out_i[c, :, :]
			for j in range(256):
				if torch.sum(pixel_mask_i[c, j, :, :]) > 0:
					temp_target_c_j = target_i[c, :, :] * pixel_mask_i[c, j, :, :]
					temp_out_c_j = out_i[c, :, :] * pixel_mask_i[c, j, :, :]
					sum_target_c_j = torch.sum(temp_target_c_j)
					sum_out_c_j = torch.sum(temp_out_c_j)
					num = torch.sum(pixel_mask_i[c, j, :, :])
					gap = sum_target_c_j/(num*1.0) - sum_out_c_j/(num*1.0)
					temp_out_c_j = temp_out_c_j + gap
					out_result_i_c += temp_out_c_j * pixel_mask_i[c, j, :, :]
			out_result_i[c, :, :] = out_result_i_c
		out_result[i] = out_result_i
		i += 1
	return out_result