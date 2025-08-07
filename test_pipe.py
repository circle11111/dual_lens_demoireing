import sys
sys.path.append("FlowFormer")
from flow_main import compute_flow # in FlowFormer/flow_main.py
import torch
import cv2
import torch.nn.functional as F

from src.network import my_model
from utils.tool import *
from utils.color_adjust import *
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Test our dual-lens demoireing pipeline by a focused-defocused image pair')
parser.add_argument('--video_name', default="20230601_154119", type=str, help='The video name in DualReal')
parser.add_argument('--num', default="0842", type=str, help='The number of frame in the video')
parser.add_argument('--pos_threshold', default=2.5, type=float, help='The threshold used in forward-backward consistency check')
parser.add_argument('--sigmaColor', default=7, type=float, help='sigmaColor used in joint bilateral filter')
parser.add_argument('--sigmaSpace', default=7, type=float, help='sigmaSpace used in joint bilateral filter')
parser.add_argument('--input_path', default="examples", type=str, help='The root of inputs')
parser.add_argument('--output_path', default="results", type=str, help='The root of outputs')
parser.add_argument('--EN_FEATURE_NUM', default=16, type=int, help='The initial channel number of dense blocks of encoder')
parser.add_argument('--EN_INTER_NUM', default=16, type=int, help='The growth rate (intermediate channel number) of dense blocks of encoder')
parser.add_argument('--DE_FEATURE_NUM', default=24, type=int, help='The initial channel number of dense blocks of decoder')
parser.add_argument('--DE_INTER_NUM', default=16, type=int, help='The growth rate (intermediate channel number) of dense blocks of decoder')
parser.add_argument('--SAM_NUMBER', default=2, type=int, help='The number of SAM for each encoder or decoder level; set 1 for our ESDNet, and 2 for ESDNet-L')
parser.add_argument('--checkpoint', default='checkpoints/G_ssim.pth', type=str, help='Checkpoint of demoireing network')
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)

model = my_model(en_feature_num=args.EN_FEATURE_NUM, en_inter_num=args.EN_INTER_NUM, de_feature_num=args.DE_FEATURE_NUM,
                de_inter_num=args.DE_INTER_NUM, sam_number=args.SAM_NUMBER)
model.load_state_dict(torch.load(args.checkpoint))
model.to(device)
model.eval()


def demoireing(image1, image2):
    b, c, h, w = image1.shape
    h_new, h_gap = find_padding(h, 64)
    w_new, w_gap = find_padding(w, 64)
    with torch.no_grad():
        if h_new > h or w_new > w:
            image1_pad = torch.zeros([b, c, h_new, w_new]).to(image1.device)
            image1_pad[..., h_gap:h_gap+h, w_gap:w_gap+w] = image1
            image2_pad = torch.zeros([b, c, h_new, w_new]).to(image2.device)
            image2_pad[..., h_gap:h_gap+h, w_gap:w_gap+w] = image2
            out_1_pad, _, _ = model(image1_pad, image2_pad)
            out_1 = out_1_pad[..., h_gap:h_gap+h, w_gap:w_gap+w]
        else:
            out_1, _, _ = model(image1, image2)     
    return out_1


def img_demoireing(image1, image2):
    img_demoire = demoireing(image1 / 255, image2 / 255)
    return img_demoire * 255


main_path = os.path.join(args.input_path, args.video_name, "main", args.num + ".png")
guide_path = os.path.join(args.input_path, args.video_name, "guide", args.num + ".png")
adjusted_path = os.path.join(args.output_path, args.video_name, "adjusted")
demoired_path = os.path.join(args.output_path, args.video_name, "demoired")
filterd_path = os.path.join(args.output_path, args.video_name, "filterd")

os.makedirs(adjusted_path, exist_ok=True)
os.makedirs(demoired_path, exist_ok=True)
os.makedirs(filterd_path, exist_ok=True)

W = 1920
H = 1080

size = [H, W]
down_size = [396, 704]
scale = H / down_size[0]

img_main = cv2.imread(main_path)      # focused image
img_guide = cv2.imread(guide_path)    # defocused image

with torch.no_grad():
    img_main = BGRarray2tensor(img_main)
    img_guide = BGRarray2tensor(img_guide)

    img_main_down = F.interpolate(img_main, size=down_size, mode='bilinear', align_corners=True)
    img_guide_down = F.interpolate(img_guide, size=down_size, mode='bilinear', align_corners=True)

    flow_1_down = compute_flow(img_main_down, img_guide_down)
    flow_2_down = compute_flow(img_guide_down, img_main_down)

    flow_1 = F.interpolate(flow_1_down, size=size, mode='bilinear', align_corners=True) * scale

    occ_down = occlusion(flow_1_down, flow_2_down, args.pos_threshold) # forward-backward consistency check
    occ = F.interpolate(occ_down.float(), size=size) > 0
    img_guide_warped = backward_warp(img_guide, flow_1)

    mask = ~occ
    img_adjusted = color_adjust(img_guide_warped, img_main, mask, batch_size=9, delta=5) # adjust the color of the defocused image to match the focused image

    img_adjusted = img_adjusted * mask + img_main * (~mask)

    cv2.imwrite(os.path.join(adjusted_path, args.num + ".png"), tensor2nBGRarray(img_adjusted)[0]) # aligned defocused image

    img_demoired = img_demoireing(img_main, img_adjusted)

    cv2.imwrite(os.path.join(demoired_path, args.num + ".png"), tensor2nBGRarray(img_demoired)[0]) # demoired image

    img_filterd = jointBilateralFilter(img_main, img_demoired, sigmaColor=args.sigmaColor, sigmaSpace=args.sigmaSpace) 

    cv2.imwrite(os.path.join(filterd_path, args.num + ".png"), tensor2nBGRarray(img_filterd)[0])   # filterd image