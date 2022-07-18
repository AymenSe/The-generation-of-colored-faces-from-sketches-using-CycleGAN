import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import yaml
from models.disc import Discriminator
from models.gen import Generator
from data import PairedCUHKDataset
from data.transforms import *
from utils import *
from trainer import Trainer
import os

def main():
    config = yaml.load(open("./config/config.yaml", "r"))
    LEARNING_RATE = config["LEARNING_RATE"]
    BATCH_SIZE = config["BATCH_SIZE"]
    START_EPOCH = config["START_EPOCH"]
    LOAD_PATH = config["LOAD_PATH"]
    DATA_PATH = config["DATA_PATH"]
    ROOT = config["ROOT"]
    NUM_WORKERS = config["NUM_WORKERS"]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    disc_sketch = Discriminator(in_channels=3).to(DEVICE) # To evaluate whether the input is a real sketch or not
    disc_photo = Discriminator(in_channels=3).to(DEVICE) # To evaluate whether the input is a real rgb or not 
    gen_sketch = Generator(img_channels=3).to(DEVICE) # To generate a Sketch fake images (input: rgb)
    gen_photo = Generator(img_channels=3).to(DEVICE) # To generate an RGB fake images (input: sketc


    ################################
    #       Optimization part      #
    ################################
    opt_disc = optim.Adam(
        list(disc_sketch.parameters()) + list(disc_photo.parameters()),
        lr = LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    opt_gen = optim.Adam(
        list(gen_sketch.parameters()) + list(gen_photo.parameters()),
        lr = LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    
    ################################
    #         Loss functions       #
    ################################
    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    ################################
    #   Load pre-trained weights   #
    ################################

    if LOAD_PATH:
        print("Loading weights from {}".format(LOAD_PATH))
        load_checkpoint(
            LOAD_PATH['gen_sketch'], gen_sketch, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            LOAD_PATH['gen_photo'], gen_photo, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            LOAD_PATH['disc_sketch'], disc_sketch, opt_disc, LEARNING_RATE,
        )
        load_checkpoint(
            LOAD_PATH['disc_photo'], disc_photo, opt_disc, LEARNING_RATE,
        )  
    else:
        print("No pre-trained weights found")
        gen_sketch = gen_sketch.apply(weights_init)
        gen_photo = gen_photo.apply(weights_init)
        disc_sketch = disc_sketch.apply(weights_init)
        disc_photo = disc_photo.apply(weights_init)
    
    ################################
    #     Data and Dataloaders     #
    ################################
    print("Loading data from {}".format(DATA_PATH))
    dataset = PairedCUHKDataset(ROOT=DATA_PATH, split_dir='train', transform=transforms)
    test_dataset = PairedCUHKDataset(ROOT=DATA_PATH, split_dir='test', transform=test_transforms)
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    ################################
    #       Training               #
    ################################

    trainer = Trainer(
        gen_sketch=gen_sketch,
        gen_photo=gen_photo,
        disc_sketch=disc_sketch,
        disc_photo=disc_photo,
        opt_gen=opt_gen,
        opt_disc=opt_disc,
        criteria=l1,
        identity_loss=mse,
        device=DEVICE,
        config=config)
    
    trainer.train(dataloader, test_dataloader, START_EPOCH, d_scaler, g_scaler)



if __name__ == '__main__':
    main()
    