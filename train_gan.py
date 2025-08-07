from torch.utils.data import DataLoader
import argparse
import torch
import os
import sys
sys.path.append('src')
from src.trainer import *
from src.network import my_model
from src.pipeline import GenerateDataset
from utils import worker_init_fn, get_path_data, get_img_info, save_data_path
from src.gan.srgan_gan_cat_2_giga import Discriminator


parser = argparse.ArgumentParser(description='Train our demoiring network and discriminator')
parser.add_argument('--dataset', default='DualSynthetic', type=str, help='dataset infomation path, including train.csv and val.csv')
parser.add_argument('--data_path', default=None, type=str, help='data path, including train, test and validation set')
parser.add_argument('--load_dir', default='checkpoints', type=str, help='path to load pre-trained models')
parser.add_argument('--G_filename', default=None, type=str, help='parameters of pre-trained demoireing network, e.g. G_*.pth')
parser.add_argument('--D_filename', default=None, type=str, help='parameters of pre-trained discriminator, e.g. D_*.pth')
parser.add_argument('--save_dir', default='checkpoints', type=str, help='path to save your models')
parser.add_argument('--epochs', default=100, type=int, help='number of train epoches')
parser.add_argument('--lr', default=0.0002, type=float, help='initial learning rate for Adam')
parser.add_argument('--train_batch_size', default=2, type=int, help='batch_size of train_dataloader')
parser.add_argument('--val_batch_size', default=2, type=int, help='batch_size of val_dataloader')
parser.add_argument('--threads', default=0, type=int, help='num_workers of dataloader')
parser.add_argument('--EN_FEATURE_NUM', default=16, type=int, help='The initial channel number of dense blocks of encoder')
parser.add_argument('--EN_INTER_NUM', default=16, type=int, help='The growth rate (intermediate channel number) of dense blocks of encoder')
parser.add_argument('--DE_FEATURE_NUM', default=24, type=int, help='The initial channel number of dense blocks of decoder')
parser.add_argument('--DE_INTER_NUM', default=16, type=int, help='The growth rate (intermediate channel number) of dense blocks of decoder')
parser.add_argument('--SAM_NUMBER', default=2, type=int, help='The number of SAM for each encoder or decoder level; set 1 for our ESDNet, and 2 for ESDNet-L')
parser.add_argument('--aug_list', default=["level_flip", "vertical_flip", "rot", "rotate"], type=list, help='ratio of val data in train dataset')
parser.add_argument('--target_size', default=(256, 256), type=int, help='image size for training')
parser.add_argument('--channels', default=3, type=int, help='channel num of the image')
parser.add_argument('--layer_num', default=3, type=int, help='layer num of the net')
parser.add_argument('--LAM', default=1, type=int, help='The loss weight for L1 loss')
parser.add_argument('--LAM_L1', default=10, type=int, help='The loss weight for consistency loss')
parser.add_argument('--LAM_P', default=1, type=int, help='The loss weight for perceptual loss')
parser.add_argument('--LAM_GAN', default=0.1, type=int, help='The loss weight for gan loss')
parser.add_argument('--LAM_FFT', default=0.1, type=int, help='The loss weight for high-frequency loss')
args = parser.parse_args()

model_G = my_model(en_feature_num=args.EN_FEATURE_NUM, en_inter_num=args.EN_INTER_NUM, de_feature_num=args.DE_FEATURE_NUM, de_inter_num=args.DE_INTER_NUM, sam_number=args.SAM_NUMBER)
model_D = Discriminator(args.target_size, args.channels, args.layer_num)

if args.load_dir is not None:
    if args.G_filename is not None:
        G_load_path = os.path.join(args.load_dir, args.G_filename)
        model_G.load_state_dict(torch.load(G_load_path))
    if args.D_filename is not None:
        D_load_path = os.path.join(args.load_dir, args.D_filename)
        model_D.load_state_dict(torch.load(D_load_path))


train_data_path = os.path.join('datasets', args.dataset, 'train.csv')
val_data_path = os.path.join('datasets', args.dataset, 'val.csv')
if not (os.path.exists(train_data_path) and os.path.exists(val_data_path)):
    os.makedirs(os.path.join('datasets', args.dataset), exist_ok=True)
    train_main_info = get_img_info(os.path.join(args.data_path, 'train', 'main'))
    train_guide_info = get_img_info(os.path.join(args.data_path, 'train', 'guide'))
    train_target_info = get_img_info(os.path.join(args.data_path, 'train', 'target'))
    val_main_info = get_img_info(os.path.join(args.data_path, 'val', 'main'))
    val_guide_info = get_img_info(os.path.join(args.data_path, 'val', 'guide'))
    val_target_info = get_img_info(os.path.join(args.data_path, 'val', 'target'))
    save_data_path(train_main_info, train_guide_info, train_target_info, train_data_path)
    save_data_path(val_main_info, val_guide_info, val_target_info, val_data_path)

train_dataset_info = get_path_data(train_data_path)
val_dataset_info = get_path_data(val_data_path)

train_dataset = GenerateDataset(train_dataset_info, args)
val_dataset = GenerateDataset(val_dataset_info, args)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.train_batch_size,
    shuffle=True,
    num_workers=args.threads,
    worker_init_fn=worker_init_fn
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.val_batch_size,
    shuffle=False,
    num_workers=args.threads,
    worker_init_fn=worker_init_fn
)

trainer = Trainer_GAN(model_G, model_D, train_dataloader, val_dataloader, args)
trainer.train()
