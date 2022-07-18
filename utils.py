import torch
import torch.nn as nn

def save_train_examples(sketch2photo_gen, photo2sketch_gen, idx, folder_to_save_trained):
    pass

def metrics(photo, sketch, fake_sketch, fake_photo, rec_photo, rec_sketch, root, epoch):
    pass

def save_image(images, path):
    pass

def save_checkpoint(model, optimizer, filename):
    pass


def load_checkpoint(path, model, optimizer, lr):
    pass

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)