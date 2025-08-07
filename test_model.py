import os
import cv2
import argparse
import warnings
from utils.tool import *
import torch
from src.network import my_model
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)

parser = argparse.ArgumentParser(description='Test our dual-lens demoireing network by a pair of pre-aligned images')
parser.add_argument('--EN_FEATURE_NUM', default=16, type=int, help='The initial channel number of dense blocks of encoder')
parser.add_argument('--EN_INTER_NUM', default=16, type=int, help='The growth rate (intermediate channel number) of dense blocks of encoder')
parser.add_argument('--DE_FEATURE_NUM', default=24, type=int, help='The initial channel number of dense blocks of decoder')
parser.add_argument('--DE_INTER_NUM', default=16, type=int, help='The growth rate (intermediate channel number) of dense blocks of decoder')
parser.add_argument('--SAM_NUMBER', default=2, type=int, help='The number of SAM for each encoder or decoder level; set 1 for our ESDNet, and 2 for ESDNet-L')
parser.add_argument('--checkpoint', default='checkpoints/G_ssim.pth', type=str, help='Checkpoint of demoireing network')
args = parser.parse_args()

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

main_path = "examples/ambush_4/main/frame_0033.png"               # focused image path
guide_path = "examples/ambush_4/guide_warped/frame_0033.png"      # defocused image path
demoired_path = "results/ambush_4/demoired/frame_0033.png"        # demoireing result path
os.makedirs("results/ambush_4/demoired", exist_ok=True)

img_main = cv2.imread(main_path)
img_guide = cv2.imread(guide_path)

img_main = BGRarray2tensor(img_main)
img_guide = BGRarray2tensor(img_guide)

img_demoired = img_demoireing(img_main, img_guide)

cv2.imwrite(demoired_path, tensor2nBGRarray(img_demoired)[0])
