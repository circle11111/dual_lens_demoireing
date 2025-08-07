import torch.nn.functional as F
import torch
import numpy as np
import math


def backward_warp(img, flow):
    b, _, h, w = img.shape
    x = torch.arange(1, w + 1).view(1, -1).repeat(h, 1).view(1, 1, h, w)
    y = torch.arange(1, h + 1).view(-1, 1).repeat(1, w).view(1, 1, h, w)
    grid = torch.cat((x, y), 1).float().cuda()
    grid = grid + flow
    grid[:, 0, :, :] = grid[:, 0, :, :] / max(w - 1, 1)
    grid[:, 1, :, :] = grid[:, 1, :, :] / max(h - 1, 1)
    grid = grid * 2.0 - 1.0
    grid = grid.permute(0, 2, 3, 1)
    img_warp = F.grid_sample(img, grid, align_corners=True)
    return img_warp


def occlusion(flow1, flow2, pos_threshold=2.5):
    b, _, h, w = flow1.shape
    x = torch.arange(1, w + 1).view(1, -1).repeat(h, 1).view(1, 1, h, w)
    y = torch.arange(1, h + 1).view(-1, 1).repeat(1, w).view(1, 1, h, w)
    grid = torch.cat((x, y), 1).float().cuda().repeat(b, 1, 1, 1)
    grid1 = backward_warp(grid, flow2)
    grid2 = backward_warp(grid1, flow1)
    pos_diff = torch.mean(torch.abs(grid - grid2), 1, True)
    pos_occ = pos_diff > pos_threshold
    return pos_occ


def getGaussianKernel(ksize, sigma):
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center) # 元素与矩阵中心的横向距离
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2)) # 计算一维卷积核
    # 根据指数函数性质，利用矩阵乘法快速计算二维卷积核
    kernel = kernel1d[..., None] @ kernel1d[None, ...] 
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum() # 归一化
    return kernel

#51,10,10
def jointBilateralFilter(input_img, guide_img, ksize=31, sigmaColor=5, sigmaSpace=5, patch_size=100):
    B, C, H, W = input_img.shape
    device = guide_img.device
    j_times, i_times = int(math.ceil(H / patch_size)), int(math.ceil(math.ceil(W / patch_size)))
    input_img_expand = input_img
    guide_img_expand = guide_img
    output_img = torch.zeros_like(input_img_expand).to(device)
    
    pad = (ksize - 1) // 2
    input_img_pad = F.pad(input_img_expand, pad=[pad, pad, pad, pad], mode='reflect')
    guide_img_pad = F.pad(guide_img_expand, pad=[pad, pad, pad, pad], mode='reflect')

    input_patches = input_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    guide_patches = guide_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)

    weights_space = getGaussianKernel(ksize, sigmaSpace).to(device)

    for y_begin in range(j_times):
        for x_begin in range(i_times):
            j_begin, j_end = y_begin * patch_size, min((y_begin + 1) * patch_size, H)
            i_begin, i_end = x_begin * patch_size, min((x_begin + 1) * patch_size, W)
            diff_color = guide_patches[:, :, j_begin:j_end, i_begin:i_end, :, :] - guide_img_expand[:, :, j_begin:j_end, i_begin:i_end].unsqueeze(-1).unsqueeze(-1)
            weights_color = torch.exp(-(diff_color ** 2) / (2 * sigmaColor ** 2))
            # weights_color /= weights_color.sum(dim=(-1, -2), keepdim=True)
            weights = weights_color * weights_space
            # weights_sum[:, :, j_begin:j_end, i_begin:i_end] = weights.sum(dim=(-1, -2))
            output_img[:, :, j_begin:j_end, i_begin:i_end] = (weights * input_patches[:, :, j_begin:j_end, i_begin:i_end, :, :]).sum(dim=(-1, -2)) / weights.sum(dim=(-1, -2))

    # return output_img[:, :, :H, :W] / torch.maximum(weights_sum[:, :, :H, :W], torch.tensor(1e-6))
    return output_img


def BGRarray2tensor(image, DEVICE='cuda'):
    image = image[:, :, ::-1].copy()
    image = torch.from_numpy(image.transpose((2, 0, 1))).float()
    return image[None].to(DEVICE)


def tensor2nBGRarray(x):
    return x.permute(0, 2, 3, 1).cpu().numpy()[:, :, :, ::-1]


def find_padding(x, k):
    y = ((x + k - 1) // k) * k
    i = (y - x) // 2
    return y, i
    

def padding_for_64x(img):
    b, c, h, w = img.shape
    h_new, h_gap = find_padding(h, 64)
    w_new, w_gap = find_padding(w, 64)
    img_pad = torch.zeros([b, c, h_new, w_new]).to(img.device)
    img_pad[..., h_gap:h_gap+h, w_gap:w_gap+w] = img
    return img_pad