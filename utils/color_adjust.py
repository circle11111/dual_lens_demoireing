import torch
import math

def rgb2ycbcr(img):
    out = torch.zeros_like(img, device=img.device) 
    img = img / 255
    r, g, b = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
    out[:, 0, :, :] = 65.481 * r + 128.553 * g + 24.966 * b + 16
    out[:, 1, :, :] = -37.797 * r - 74.203 * g + 112 * b + 128
    out[:, 2, :, :] = 112 * r - 93.786 * g - 18.214 * b + 128
    return out


def ycbcr2rgb(img):  
    out = torch.zeros_like(img, device=img.device)
    y, cb, cr = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
    y = 1.1644 * (y - 16)
    cb = cb - 128
    cr = cr - 128
    out[:, 0, :, :] = y + 1.596 * cr
    out[:, 1, :, :] = y - 0.3918 * cb - 0.813 * cr
    out[:, 2, :, :] = y + 2.0172 * cb
    return out


def get_weight(win_size, stride, device):
    x = torch.arange(0, win_size).to(device).view(1, -1).repeat(win_size, 1).view(1, 1, win_size, win_size)
    y = torch.arange(0, win_size).to(device).view(-1, 1).repeat(1, win_size).view(1, 1, win_size, win_size)
    dy = torch.minimum(y, win_size - 1 - y)
    dx = torch.minimum(x, win_size - 1 - x)
    d = torch.minimum(dx, dy) + 1
    weight = torch.minimum(d, torch.tensor(win_size - stride).to(device)) / (win_size - stride)
    return weight


def adjust_by_offset(img1, img2, valid_region, delta=3):
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    offset = img2 - img1
    bin = torch.arange(0, 256, device=img1.device).view(-1, 1, 1, 1, 1)
    mask = (valid_region != 0) & (img1 - bin >= - delta) & (img1 - bin <= delta)
    count = torch.maximum(torch.sum(mask, [3, 4], True), torch.tensor(1).to(img1.device))  
    out = (img1.round() == bin) * (img1 + torch.sum(offset * mask, [3, 4], True) / count)
    return torch.sum(out, 0)


def color_adjust(img1, img2, valid_region, win_size=400, stride=360, delta=5, batch_size=3):
    b, c, h, w = img1.shape
    h_pad = int(math.ceil((h - win_size) / stride) * stride + win_size)
    w_pad = int(math.ceil((w - win_size) / stride) * stride + win_size)
    img1_pad = torch.zeros([b, c, h_pad, w_pad]).to(img1.device)
    img2_pad = torch.zeros([b, c, h_pad, w_pad]).to(img1.device)
    valid_region_pad = torch.zeros([b, 1, h_pad, w_pad]).to(img1.device)
    img1_pad[:, :, :h, :w] = rgb2ycbcr(img1)
    img2_pad[:, :, :h, :w] = rgb2ycbcr(img2)
    valid_region_pad[:, :, :h, :w] = valid_region
    patch_weight = get_weight(win_size, stride, img1.device)
    img1_adjust = torch.zeros([b, c, h_pad, w_pad]).to(img1.device)
    cnt = torch.zeros([b, c, h_pad, w_pad]).to(img1.device)
    j_num = (h_pad - win_size) // stride + 1
    i_num = (w_pad - win_size) // stride + 1
    j_list = torch.arange(0, j_num, dtype=torch.int64).view(1, -1).repeat(i_num, 1).flatten()
    i_list = torch.arange(0, i_num, dtype=torch.int64).view(-1, 1).repeat(1, j_num).flatten()
    idx_begin = 0
    length = len(j_list)
    batch_size = min(batch_size, length)
    while(idx_begin < length):
        img1_patch = torch.tensor([]).to(img1.device)
        img2_patch = torch.tensor([]).to(img1.device)
        valid_region_patch = torch.tensor([]).to(img1.device)
        idx_end = min(length, idx_begin + batch_size)
        for idx in range(idx_begin, idx_end):
            img1_patch = torch.cat([img1_patch, img1_pad[:, :, j_list[idx] * stride:j_list[idx] * stride + win_size, i_list[idx] * stride:i_list[idx] * stride + win_size]])
            img2_patch = torch.cat([img2_patch, img2_pad[:, :, j_list[idx] * stride:j_list[idx] * stride + win_size, i_list[idx] * stride:i_list[idx] * stride + win_size]])
            valid_region_patch = torch.cat([valid_region_patch, valid_region_pad[:, :, j_list[idx] * stride:j_list[idx] * stride + win_size, i_list[idx] * stride:i_list[idx] * stride + win_size]]) 
        img1_patch = adjust_by_offset(img1_patch, img2_patch, valid_region_patch, delta=delta)
        img1_patch = ycbcr2rgb(img1_patch) * valid_region_patch * patch_weight
        for idx in range(idx_begin, idx_end):
            img1_adjust[:, :, j_list[idx] * stride:j_list[idx] * stride + win_size, i_list[idx] * stride:i_list[idx] * stride + win_size] += img1_patch[idx - idx_begin:idx - idx_begin + 1, :, :, :]
            cnt[:, :, j_list[idx] * stride:j_list[idx] * stride + win_size, i_list[idx] * stride:i_list[idx] * stride + win_size] += patch_weight
        idx_begin += batch_size
    return img1_adjust[:, :, :h, :w] / cnt[:, :, :h, :w]