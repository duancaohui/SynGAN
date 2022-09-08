#!/usr/bin/python3

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import torch
import torchvision
from .utils import LambdaLR, Logger, ReplayBuffer, norm01
from .utils import weights_init_normal, get_config
import time
from .datasets import ImageDataset7T, ValDataset, TestDataset
from Model.CycleGan import *
from Model.unet import *
from Model.Unit_self import Unet_tf
from .utils import Resize, ToTensor, smooothing_loss, augment_Rigid
from .utils import Logger
from .reg import Reg
from torchvision.transforms import RandomAffine, ToPILImage
from torch.utils.tensorboard import SummaryWriter
from .transformer import Transformer_2D
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.image as mpimg
import cv2
import tensorflow as tf
import scipy.io


class SynGAN_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config

        # def networks
        self.netG_A2B = Unet_tf(config['input_nc'], config['output_nc']).cuda()

        # self.netG_A2B = Generator(config['input_nc'], config['output_nc']).cuda()
        self.netG_A2B.apply(self.weight_init)

        self.netD_B = Discriminator(config['input_nc']*2).cuda()
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
                
        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size_x'], config['size_y'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size_x'], config['size_y'])
        self.input_M = Tensor(config['batchSize'], config['output_nc'], config['size_x'], config['size_y'])

        self.target_real = Variable(Tensor(config['batchSize'], 1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(config['batchSize'], 1).fill_(0.0), requires_grad=False)

        # self.fake_A_buffer = ReplayBuffer()
        # self.fake_B_buffer = ReplayBuffer()

        #Dataset loader
        level = config['noise_level']
        transforms_1 = [ToTensor(), Resize(size_tuple=(config['size_x'], config['size_y']))]

        self.dataloader = DataLoader(ImageDataset7T(config['dataroot'], transforms_1=transforms_1),
                                batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'])

        val_transforms = [ToTensor(), Resize(size_tuple=(config['size_x'], config['size_y']))]
        
        self.val_data = DataLoader(ValDataset(config['val_dataroot'], transforms_=val_transforms, unaligned=False),
                                batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])

        self.test_data = DataLoader(TestDataset(config['val_dataroot'], transforms_=val_transforms, unaligned=False),
                                   batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])

       # Loss plot
        self.logger = Logger(config['name'], config['port'], config['n_epochs'], len(self.dataloader))

        if config['train']:
            self.summary_directory = os.path.join(self.config['save_root'], self.config['summary_writer'])
            self.summary_writer = SummaryWriter(self.summary_directory)
        else:
            self.summary_directory = os.path.join(self.config['save_root'], self.config['test_summary'])
            self.summary_writer = SummaryWriter(self.summary_directory)

    def train_summary(self):
        # Training
        device = torch.device('cuda:0')
        total_iter = 0

        if not os.path.exists(self.config["save_root"]):
            os.makedirs(self.config["save_root"])

        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            loss_G = 0.0
            loss_D = 0.0

            for i, batch in enumerate(self.dataloader):

                total_iter += 1

                # batch = [batch[x].to(device, non_blocking=True) for x in batch.keys()]
                # with torch.no_grad():
                #     batch = augment_Rigid(batch)

                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                real_M = Variable(self.input_M.copy_(batch['M']))

                # real_A = Variable(self.input_A.copy_(batch[0]))
                # real_B = Variable(self.input_B.copy_(batch[1]))
                # real_M = Variable(self.input_M.copy_(batch[2]))

                self.optimizer_G.zero_grad()
                fake_B = self.netG_A2B(real_A)
                loss_L1 = self.L1_loss(fake_B, real_B) * self.config['P2P_lamda']

                # gan loss:
                fake_AB = torch.cat((real_A, fake_B*real_M), 1)
                pred_fake = self.netD_B(fake_AB)
                loss_GAN_A2B = self.MSE_loss(pred_fake, self.target_real) * self.config['Adv_lamda']

                # Total loss
                toal_loss = loss_L1 + loss_GAN_A2B
                # toal_loss = loss_L1
                loss_G += toal_loss

                toal_loss.backward()
                self.optimizer_G.step()

                self.optimizer_D_B.zero_grad()
                with torch.no_grad():
                    fake_B = self.netG_A2B(real_A)

                pred_fake0 = self.netD_B(torch.cat((real_A, fake_B*real_M), 1)) * self.config['Adv_lamda']
                pred_real = self.netD_B(torch.cat((real_A, real_B), 1)) * self.config['Adv_lamda']
                loss_D_B = self.MSE_loss(pred_fake0, self.target_fake) + self.MSE_loss(pred_real, self.target_real)

                loss_D += loss_D_B
                loss_D_B.backward()
                # loss_D_B = toal_loss
                self.optimizer_D_B.step()
                self.logger.log({'loss_D_B': loss_D_B, 'loss_G': toal_loss, },
                                images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B})  # ,'SR':SysRegist_A2B

                # summary_image = torch.cat((norm01(real_A), norm01(real_B), norm01(fake_B)), -1)
                summary_image = torch.cat((real_A, real_B, fake_B), -2)

                if i % 100 == 0:

                    self.summary_writer.add_images('traning recon',
                                                   summary_image,
                                                   global_step=total_iter,
                                                   dataformats='NCHW')

            self.summary_writer.add_scalar('loss_G_epoch',
                                           loss_G / len(self.dataloader),
                                           epoch)
            self.summary_writer.add_scalar('loss_D_epoch',
                                           loss_D / len(self.dataloader),
                                           epoch)
            # Save models checkpoints
            # save checkpoint
            if (epoch + 1) % 9 == 0:
                checkpoint_name = os.path.join(self.config['save_checkpoint'], 'model_{}.pt'.format(epoch))
                torch.save(self.netG_A2B.state_dict(), checkpoint_name)

            # val
            with torch.no_grad():
                MAE = 0
                num = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                    fake_B = self.netG_A2B(real_A).detach().cpu().numpy().squeeze()
                    mae = self.MAE(fake_B, real_B)
                    MAE += mae
                    num += 1

                print('MAE:', MAE / num)

    def test(self):
        self.netG_A2B.load_state_dict(torch.load(os.path.join(self.config['save_root'], 'checkpoint', 'model_80.pt')))
        CONCAT_PATH = os.path.join(self.config['save_root'], 'concat')

        steps = len(self.test_data)  #//self.config['batchSize']
        width = self.config['size_x']
        length = self.config['size_y']

        syns = np.zeros([steps, self.config['batchSize'], 1, width, length])
        img_7Ts = np.zeros([steps, self.config['batchSize'], 1, width, length])
        img_3Ts = np.zeros([steps, self.config['batchSize'], 1, width, length])
        masks = np.zeros([steps, self.config['batchSize'], 1, width, length])

        if not os.path.exists(CONCAT_PATH):
            os.makedirs(CONCAT_PATH)

        Metrics_PATH = os.path.join(self.config['save_root'], 'metrics.mat')
        with torch.no_grad():
                MAE = 0
                PSNR = 0
                SSIM = 0
                num = 0
                
                for i, batch in enumerate(self.test_data):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    real_B = Variable(self.input_B.copy_(batch['B']))
                    real_M = Variable(self.input_M.copy_(batch['M']))

                    fake_B = self.netG_A2B(real_A)
                    max_samples = real_A.shape[0]

                    syns[i] = fake_B.cpu().numpy()
                    img_3Ts[i] = real_A.cpu().numpy()
                    img_7Ts[i] = real_B.cpu().numpy()
                    masks[i] = real_M.cpu().numpy()

                    # summary_image = torch.cat((norm01(real_A), norm01(real_B), norm01(fake_B)), -1)
                    summary_image = torch.cat((real_A, real_B, fake_B), -1)
                    self.summary_writer.add_images('traning recon',
                                                   summary_image,
                                                   global_step=i,
                                                   dataformats='NCHW')

                    summary_image = summary_image.cpu().numpy().squeeze()
                    image = tf.concat(axis=0, values=[summary_image[i] for i in range(max_samples)])

                    mpimg.imsave(os.path.join(CONCAT_PATH, '{:03d}.tiff'.format(i)), image, cmap='gray')

                    fake_B = fake_B.detach().cpu().numpy().squeeze()
                    real_B = real_B.detach().cpu().numpy().squeeze()

                    mae = self.MAE(fake_B, real_B)
                    # psnr = self.PSNR(fake_B, real_B)
                    # ssim = ssim(fake_B, real_B)
                    MAE += mae
                    # PSNR += psnr
                    # SSIM += ssim
                    num += 1
                print('MAE:', MAE/num)

                syns = np.reshape(syns, [-1, 1, width, length])[:]
                img_3Ts = np.reshape(img_3Ts, [-1, 1, width, length])[:]
                img_7Ts = np.reshape(img_7Ts, [-1, 1, width, length])[:]
                masks = np.reshape(masks, [-1, 1, width, length])[:]

                syns = np.rollaxis(np.squeeze(syns), 0, 3)
                img_3Ts = np.rollaxis(np.squeeze(img_3Ts), 0, 3)
                img_7Ts = np.rollaxis(np.squeeze(img_7Ts), 0, 3)
                masks = np.rollaxis(np.squeeze(masks), 0, 3)

                scipy.io.savemat(Metrics_PATH, {'syns': syns, 'img_3Ts': img_3Ts, 'img_7Ts': img_7Ts, 'masks': masks})
                # print('PSNR:', PSNR/num)
                # print('SSIM:', SSIM/num)
                #

    def test_time(self):
        self.netG_A2B.load_state_dict(torch.load(os.path.join(self.config['save_root'], 'checkpoint', 'model_80.pt')))

        steps = len(self.test_data)  # //self.config['batchSize']
        start = time.time()

        with torch.no_grad():
            for i, batch in enumerate(self.test_data):
                real_A = Variable(self.input_A.copy_(batch['A']))
                fake_B = self.netG_A2B(real_A)

        print('Time taken for epoch {} is {} sec\n'.format(180 + 1, time.time() - start))
