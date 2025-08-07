import torch
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as F
from torch.autograd import Variable
from src.loss import multi_VGGPerceptualLoss
from utils import PSNR_SSIM_LPIPS, log
import numpy as np
import os


cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
device = torch.device("cuda" if cuda else "cpu")


def save(model, path):
    torch.save(model.state_dict(), path)


class Trainer():
    def __init__(self, model_G, train_loader, val_loader, args):        
        self.model_G = model_G
        self.model_G.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_G = Adam(self.model_G.parameters(), lr=args.lr, betas=(0.9, 0.999))
        self.criterion = multi_VGGPerceptualLoss(lam=args.LAM, lam_p=args.LAM_P).to(device)
        self.criterion_content = L1Loss().to(device)
        self.args = args
        os.makedirs(os.path.join(args.save_dir, 'inf_psnr'), exist_ok=True)


    def train_single_step(self, img_main, img_guide, img_target):    
        self.optimizer_G.zero_grad()
        G_out1, G_out2, G_out3 = self.model_G(img_main, img_guide)
        self.metrics.compute(G_out1, img_target)
        
        loss_vgg = self.criterion(G_out1, G_out2, G_out3, img_target)
        
        loss_content = self.criterion_content(G_out1, img_target)
        loss_fft = self.criterion_content(torch.fft.fft2(G_out1), torch.fft.fft2(img_target))
        loss_G = self.args.LAM_L1 * loss_content + loss_vgg + self.args.LAM_FFT * loss_fft
        loss_G.backward()
        self.optimizer_G.step()
        

    def train_single_epoch(self):
        self.metrics = PSNR_SSIM_LPIPS()
        for index, data in enumerate(self.train_loader, start=0):
            img_main, img_guide, img_target = data
            self.train_single_step(img_main.to(device), img_guide.to(device), img_target.to(device))
        log()
        

    @torch.no_grad()
    def test_single_epoch(self):
        metrics = PSNR_SSIM_LPIPS()
        for index, data in enumerate(self.val_loader, start=0):
            img_main, img_guide, img_target = data
            out_1, out_2, out_3 = self.model_G(img_main.to(device), img_guide.to(device))
            metrics.compute(out_1, img_target.to(device))
        return metrics


    def train(self):
        log()
        best_psnr_train, best_ssim_train, best_lpips_train = [0.0, 0.0, 100.0]
        best_psnr_val, best_ssim_val, best_lpips_val = [0.0, 0.0, 100.0] 
        best_psnr_train_epoch, best_ssim_train_epoch, best_lpips_train_epoch = [0, 0, 0]
        best_psnr_val_epoch, best_ssim_val_epoch, best_lpips_val_epoch = [0, 0, 0]
        step, step_stage2 = [0, 0]
        save_dir = self.args.save_dir
        lr = self.args.lr
        for epoch in range(self.args.epochs):
            for param_group in self.optimizer_G.param_groups:
                if param_group["lr"] > 0.0001:
                    current_lr = lr / 2 ** (step / 50000)
                else:
                    if step_stage2 == 0:
                        lr = current_lr
                        step_stage2 = step_stage2 + 1
                    current_lr = lr / 2 ** (step_stage2 / 120000)
                param_group["lr"] = current_lr
            print("Epoch = {}, lr = {}".format(epoch + 1, self.optimizer_G.param_groups[0]["lr"]))
            self.model_G.train()
            self.train_single_epoch()
            psnr_train, ssim_train, lpips_train = self.metrics.get_info()
            if psnr_train > best_psnr_train:
                best_psnr_train = psnr_train
                best_psnr_train_epoch = epoch + 1
            if ssim_train > best_ssim_train:
                best_ssim_train = ssim_train
                best_ssim_train_epoch = epoch + 1
            if lpips_train < best_lpips_train:
                best_lpips_train = lpips_train
                best_lpips_train_epoch = epoch + 1
            log()
            print('train epoch = %04d , best_psnr_train_epoch = %04d , best_train_psnr= %4.4f, best_ssim_train_epoch = %04d , best_train_ssim= %4.4f, best_lpips_train_epoch = %04d , best_train_lpips= %4.4f, current_average_psnr = %4.4f , current_average_ssim = %4.4f, current_average_lpips = %4.4f' % (
                epoch + 1, best_psnr_train_epoch, best_psnr_train, best_ssim_train_epoch, best_ssim_train, best_lpips_train_epoch, best_lpips_train, psnr_train, ssim_train, lpips_train))

            self.model_G.eval()
            metrics = self.test_single_epoch()
            psnr_val, ssim_val, lpips_val = metrics.get_info()
            if psnr_val > 100:
                if ssim_val > best_ssim_val:
                    best_ssim_val = ssim_val
                    best_ssim_val_epoch = epoch + 1
                    save(self.model_G, os.path.join(save_dir, 'inf_psnr', 'G_ssim.pth'))
                if lpips_val < best_lpips_val:
                    best_lpips_val = lpips_val
                    best_lpips_val_epoch = epoch + 1
                    save(self.model_G, os.path.join(save_dir, 'inf_psnr', 'G_lpips.pth'))
            else:
                if psnr_val > best_psnr_val:
                    best_psnr_val = psnr_val
                    best_psnr_val_epoch = epoch + 1
                    save(self.model_G, os.path.join(save_dir, 'G_psnr.pth'))
                if ssim_val > best_ssim_val:
                    best_ssim_val = ssim_val
                    best_ssim_val_epoch = epoch + 1
                    save(self.model_G, os.path.join(save_dir, 'G_ssim.pth'))
                if lpips_val < best_lpips_val:
                    best_lpips_val = lpips_val
                    best_lpips_val_epoch = epoch + 1
                    save(self.model_G, os.path.join(save_dir, 'G_lpips.pth'))
            log()
            print('validation epoch = %04d , best_psnr_val_epoch = %04d , best_val_psnr = %4.4f , best_ssim_val_epoch = %04d , best_val_ssim = %4.4f, best_lpips_val_epoch = %04d , best_val_lpips = %4.4f , current_average_psnr = %4.4f , current_average_ssim = %4.4f, current_average_lpips = %4.4f' % (
                epoch + 1, best_psnr_val_epoch, best_psnr_val, best_ssim_val_epoch, best_ssim_val, best_lpips_val_epoch, best_lpips_val, psnr_val, ssim_val, lpips_val))
            

class Trainer_GAN(Trainer):
    def __init__(self, model_G, model_D, train_loader, val_loader, args):        
        super().__init__(model_G, train_loader, val_loader, args)
        self.model_D = model_D
        self.model_D.to(device)
        self.optimizer_D = Adam(self.model_D.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.criterion_GAN = MSELoss().to(device)


    def train_single_step(self, img_main, img_guide, img_target):    
        img_guide_d_2 = F.interpolate(img_guide, scale_factor = 0.5)
        img_guide_d_4 = F.interpolate(img_guide, scale_factor = 0.25)
        guide_multiscale = [img_guide, img_guide_d_2, img_guide_d_4]

        img_target_d_2 = F.interpolate(img_target, scale_factor = 0.5)
        img_target_d_4 = F.interpolate(img_target, scale_factor = 0.25)
        target_multiscale = [img_target, img_target_d_2, img_target_d_4]

        self.optimizer_G.zero_grad()
        G_out1, G_out2, G_out3 = self.model_G(img_main, img_guide)
        self.metrics.compute(G_out1, img_target)
        D_out1, D_out2, D_out3 = self.model_D([G_out1, G_out2, G_out3], guide_multiscale)
        valid1 = Variable(Tensor(np.ones(D_out1[0].shape)), requires_grad=False)
        valid2 = Variable(Tensor(np.ones(D_out1[1].shape)), requires_grad=False)
        valid3 = Variable(Tensor(np.ones(D_out1[2].shape)), requires_grad=False)
        fake1 = Variable(Tensor(np.zeros(D_out1[0].shape)), requires_grad=False)
        fake2 = Variable(Tensor(np.zeros(D_out1[1].shape)), requires_grad=False)
        fake3 = Variable(Tensor(np.zeros(D_out1[2].shape)), requires_grad=False)
        loss_vgg = self.criterion(G_out1, G_out2, G_out3, img_target)
        loss_GAN = (self.criterion_GAN(D_out1[0], valid1) + self.criterion_GAN(D_out1[1], valid2) + self.criterion_GAN(D_out1[2], valid3) + 
                    self.criterion_GAN(D_out2[0], valid2) + self.criterion_GAN(D_out2[1], valid3) + self.criterion_GAN(D_out3[0], valid3)) / 6.0
        loss_content = self.criterion_content(G_out1, img_target)
        loss_fft = self.criterion_content(torch.fft.fft2(G_out1), torch.fft.fft2(img_target))
        loss_G = self.args.LAM_L1 * loss_content + loss_vgg + self.args.LAM_GAN * loss_GAN + self.args.LAM_FFT * loss_fft
        loss_G.backward()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        D_out1, D_out2, D_out3 = self.model_D(target_multiscale, guide_multiscale)
        loss_real = (self.criterion_GAN(D_out1[0], valid1) + self.criterion_GAN(D_out1[1], valid2) + self.criterion_GAN(D_out1[2], valid3) + 
                     self.criterion_GAN(D_out2[0], valid2) + self.criterion_GAN(D_out2[1], valid3) + self.criterion_GAN(D_out3[0], valid3)) / 6.0
        D_out1, D_out2, D_out3 = self.model_D([G_out1.detach(), G_out2.detach(), G_out3.detach()], guide_multiscale)
        loss_fake = (self.criterion_GAN(D_out1[0], fake1) + self.criterion_GAN(D_out1[1], fake2) + self.criterion_GAN(D_out1[2], fake3) + 
                     self.criterion_GAN(D_out2[0], fake2) + self.criterion_GAN(D_out2[1], fake3) + self.criterion_GAN(D_out3[0], fake3)) / 6.0
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        self.optimizer_D.step()
       

    def train(self):
        log()
        best_psnr_train, best_ssim_train, best_lpips_train = [0.0, 0.0, 100.0]
        best_psnr_val, best_ssim_val, best_lpips_val = [0.0, 0.0, 100.0] 
        best_psnr_train_epoch, best_ssim_train_epoch, best_lpips_train_epoch = [0, 0, 0]
        best_psnr_val_epoch, best_ssim_val_epoch, best_lpips_val_epoch = [0, 0, 0]
        step, step_stage2 = [0, 0]
        save_dir = self.args.save_dir
        lr = self.args.lr
        for epoch in range(self.args.epochs):
            for param_group in self.optimizer_G.param_groups:
                if param_group["lr"] > 0.0001:
                    current_lr = lr / 2 ** (step / 50000)
                else:
                    if step_stage2 == 0:
                        lr = current_lr
                        step_stage2 = step_stage2 + 1
                    current_lr = lr / 2 ** (step_stage2 / 120000)
                param_group["lr"] = current_lr
            for param_group in self.optimizer_D.param_groups:
                param_group["lr"] = current_lr
            print("Epoch = {}, lr = {}".format(epoch + 1, self.optimizer_G.param_groups[0]["lr"]))
            self.model_G.train()
            self.model_D.train()
            self.train_single_epoch()
            psnr_train, ssim_train, lpips_train = self.metrics.get_info()
            if psnr_train > best_psnr_train:
                best_psnr_train = psnr_train
                best_psnr_train_epoch = epoch + 1
            if ssim_train > best_ssim_train:
                best_ssim_train = ssim_train
                best_ssim_train_epoch = epoch + 1
            if lpips_train < best_lpips_train:
                best_lpips_train = lpips_train
                best_lpips_train_epoch = epoch + 1
            log()
            print('train epoch = %04d , best_psnr_train_epoch = %04d , best_train_psnr= %4.4f, best_ssim_train_epoch = %04d , best_train_ssim= %4.4f, best_lpips_train_epoch = %04d , best_train_lpips= %4.4f, current_average_psnr = %4.4f , current_average_ssim = %4.4f, current_average_lpips = %4.4f' % (
                epoch + 1, best_psnr_train_epoch, best_psnr_train, best_ssim_train_epoch, best_ssim_train, best_lpips_train_epoch, best_lpips_train, psnr_train, ssim_train, lpips_train))

            self.model_G.eval()
            self.model_D.eval()
            metrics = self.test_single_epoch()
            psnr_val, ssim_val, lpips_val = metrics.get_info()
            if psnr_val > 100:
                if ssim_val > best_ssim_val:
                    best_ssim_val = ssim_val
                    best_ssim_val_epoch = epoch + 1
                    save(self.model_G, os.path.join(save_dir, 'inf_psnr', 'G_ssim.pth'))
                    save(self.model_D, os.path.join(save_dir, 'inf_psnr', 'D_ssim.pth'))
                if lpips_val < best_lpips_val:
                    best_lpips_val = lpips_val
                    best_lpips_val_epoch = epoch + 1
                    save(self.model_G, os.path.join(save_dir, 'inf_psnr', 'G_lpips.pth'))
                    save(self.model_D, os.path.join(save_dir, 'inf_psnr', 'D_lpips.pth'))
            else:
                if psnr_val > best_psnr_val:
                    best_psnr_val = psnr_val
                    best_psnr_val_epoch = epoch + 1
                    save(self.model_G, os.path.join(save_dir, 'G_psnr.pth'))
                    save(self.model_D, os.path.join(save_dir, 'D_psnr.pth'))
                if ssim_val > best_ssim_val:
                    best_ssim_val = ssim_val
                    best_ssim_val_epoch = epoch + 1
                    save(self.model_G, os.path.join(save_dir, 'G_ssim.pth'))
                    save(self.model_D, os.path.join(save_dir, 'D_ssim.pth'))
                if lpips_val < best_lpips_val:
                    best_lpips_val = lpips_val
                    best_lpips_val_epoch = epoch + 1
                    save(self.model_G, os.path.join(save_dir, 'G_lpips.pth'))
                    save(self.model_D, os.path.join(save_dir, 'D_lpips.pth'))
            log()
            print('validation epoch = %04d , best_psnr_val_epoch = %04d , best_val_psnr = %4.4f , best_ssim_val_epoch = %04d , best_val_ssim = %4.4f, best_lpips_val_epoch = %04d , best_val_lpips = %4.4f , current_average_psnr = %4.4f , current_average_ssim = %4.4f, current_average_lpips = %4.4f' % (
                epoch + 1, best_psnr_val_epoch, best_psnr_val, best_ssim_val_epoch, best_ssim_val, best_lpips_val_epoch, best_lpips_val, psnr_val, ssim_val, lpips_val))
            

    def train_without_validation(self):
        log()
        best_psnr_train, best_ssim_train, best_lpips_train = [0.0, 0.0, 100.0]
        best_psnr_train_epoch, best_ssim_train_epoch, best_lpips_train_epoch = [0, 0, 0]
        step, step_stage2 = [0, 0]
        save_dir = self.args.save_dir
        lr = self.args.lr
        for epoch in range(self.args.epochs):
            for param_group in self.optimizer_G.param_groups:
                if param_group["lr"] > 0.0001:
                    current_lr = lr / 2 ** (step / 50000)
                else:
                    if step_stage2 == 0:
                        lr = current_lr
                        step_stage2 = step_stage2 + 1
                    current_lr = lr / 2 ** (step_stage2 / 120000)
                param_group["lr"] = current_lr
            for param_group in self.optimizer_D.param_groups:
                param_group["lr"] = current_lr
            print("Epoch = {}, lr = {}".format(epoch + 1, self.optimizer_G.param_groups[0]["lr"]))
            self.model_G.train()
            self.model_D.train()
            self.train_single_epoch()
            psnr_train, ssim_train, lpips_train = self.metrics.get_info()
            if psnr_train > best_psnr_train:
                best_psnr_train = psnr_train
                best_psnr_train_epoch = epoch + 1
                save(self.model_G, os.path.join(save_dir, 'G_psnr.pth'))
                save(self.model_D, os.path.join(save_dir, 'D_psnr.pth'))
            if ssim_train > best_ssim_train:
                best_ssim_train = ssim_train
                best_ssim_train_epoch = epoch + 1
                save(self.model_G, os.path.join(save_dir, 'G_ssim.pth'))
                save(self.model_D, os.path.join(save_dir, 'D_ssim.pth'))
            if lpips_train < best_lpips_train:
                best_lpips_train = lpips_train
                best_lpips_train_epoch = epoch + 1
                save(self.model_G, os.path.join(save_dir, 'G_lpips.pth'))
                save(self.model_D, os.path.join(save_dir, 'D_lpips.pth'))
            log()
            print('train epoch = %04d , best_psnr_train_epoch = %04d , best_train_psnr= %4.4f, best_ssim_train_epoch = %04d , best_train_ssim= %4.4f, best_lpips_train_epoch = %04d , best_train_lpips= %4.4f, current_average_psnr = %4.4f , current_average_ssim = %4.4f, current_average_lpips = %4.4f' % (
                epoch + 1, best_psnr_train_epoch, best_psnr_train, best_ssim_train_epoch, best_ssim_train, best_lpips_train_epoch, best_lpips_train, psnr_train, ssim_train, lpips_train))
