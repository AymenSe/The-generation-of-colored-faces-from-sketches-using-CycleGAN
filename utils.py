import torch
import torch.nn as nn
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import json

def save_train_examples(sketch2photo_gen, photo2sketch_gen, idx, folder_to_save_trained):
    sketch2photo_gen = torch.cat(sketch2photo_gen, dim=3)
    photo2sketch_gen = torch.cat(photo2sketch_gen, dim=3)

    photo2sketch_gen_path = os.path.join(folder_to_save_trained, f"photo2sketch_{idx}.png")
    sketch2photo_gen_path = os.path.join(folder_to_save_trained, f"sketch2photo_{idx}.png")
    
    save_image(sketch2photo_gen.data, sketch2photo_gen_path)
    save_image(photo2sketch_gen.data, photo2sketch_gen_path)

def metrics(photo, sketch, fake_sketch, fake_photo, rec_photo, rec_sketch, root, epoch):
    pass


def save_checkpoint(model, optimizer, filename):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def vis_losses(root):
    
    with open(root + "/gen_loss.json", 'r') as g_l:
        gen_loss_arr = json.load(g_l)
    
    with open(root + "/disc_loss.json", 'r') as d_l:
        disc_loss_arr = json.load(d_l)

    assert len(gen_loss_arr) == len(disc_loss_arr), "Error arrays lenghts are not equal"
    epochs = len(gen_loss_arr)

    arr_losses_1 = np.array(gen_loss_arr)
    arr_losses_2 = np.array(disc_loss_arr)

    # Plotting
    plt.plot(arr_losses_1, label='Gen loss')  
    plt.plot(arr_losses_2, label='Disc Loss')
    plt.xlabel(f'epochs = {epochs}')
    plt.ylabel('losses')
    plt.title("Gen and Disc Losses")
    plt.legend()
    plt.savefig(root + '/losses.png')