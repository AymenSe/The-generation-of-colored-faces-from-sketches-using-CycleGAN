from tqdm import tqdm
import os
import json
from torchvision.utils import save_image

import torch
import torch.nn as nn

from utils import *

class CycleGANTrainer:
    def __init__(self, gen_sketch, gen_photo, disc_sketch, disc_photo, optim_gen, optim_disc, criteria, identity_loss, device, **params) -> None:
        self.genAtoB = gen_sketch
        self.genBtoA = gen_photo
        self.discA = disc_photo
        self.discB = disc_sketch
        self.opt_disc = optim_disc
        self.opt_gen = optim_gen
        self.criteria = criteria
        self.identity_loss = identity_loss
        self.device = device
        
        self.folder_to_save_trained = params["folder_to_save_trained"]
        self.LAMBDA_CYCLE = params["LAMBDA_CYCLE"]
        self.LAMBDA_IDT = params["LAMBDA_IDT"]
        self.print_every = params["print_every"]
        self.ROOT = params["ROOT"]
        self.EPOCHS = params["EPOCHS"]
        self.SAVE_MODEL = params["SAVE_MODEL"]



    def train(self, dataloader, start_epoch, d_scaler, g_scaler):
        """
        Train the model for a given number of self.EPOCHS.
        Args:
            dataloader: Dataloader for the training set.
            start_epoch: Epoch to start training from.
            device: Device to use for training.
            
        """
        # train mode.
        self.genAtoB.train()
        self.genBtoA.train()
        self.discA.train()
        self.discB.train()

        with open(self.ROOT+"/gen_loss.json", 'r') as g_l:
            gen_loss_arr = json.load(g_l)
    
        with open(self.ROOT+"/disc_loss.json", 'r') as d_l:
            disc_loss_arr = json.load(d_l)

        
        for epoch in range(start_epoch, self.EPOCHS + start_epoch):
            

            loop = tqdm(dataloader, leave=True)
            
            for idx, (photo, sketch) in enumerate(loop):
                photo = photo.to(self.device)
                sketch = sketch.to(self.device)
                # print(photo.shape, sketch.shape)
                # train discriminator
                with torch.cuda.amp.autocast():
                    fake_sketch = self.genAtoB(photo)
                    fake_photo = self.genBtoA(sketch)

                    # print(fake_sketch.shape, sketch.shape, fake_photo.shape, photo.shape)
                    fake_sketch_loss = self.criteria(self.discB(fake_sketch.detach()), torch.zeros_like(fake_sketch).to(self.device))
                    real_sketch_loss = self.criteria(self.discB(sketch), torch.ones_like(sketch).to(self.device))
                    fake_photo_loss = self.criteria(self.discA(fake_photo.detach()), torch.zeros_like(fake_photo).to(self.device))
                    real_photo_loss = self.criteria(self.discA(photo), torch.ones_like(photo).to(self.device))
                    disc_loss = (fake_sketch_loss + real_sketch_loss + fake_photo_loss + real_photo_loss) / 4
                
                self.opt_disc.zero_grad()
                d_scaler.scale(disc_loss).backward()                
                d_scaler.step(self.opt_disc)
                d_scaler.update()


                # train generator
                with torch.cuda.amp.autocast():
                    fake_sketch = self.genAtoB(photo)
                    fake_photo = self.genBtoA(fake_sketch)
                    fake_sketch_loss = self.criteria(self.discB(fake_sketch), torch.ones_like(fake_sketch).to(self.device))
                    fake_photo_loss = self.criteria(self.discA(fake_photo), torch.ones_like(fake_photo).to(self.device))

                    # Cycle loss
                    rec_photo = self.genBtoA(fake_sketch)
                    rec_sketch = self.genAtoB(fake_photo)
                    cycle_photo_loss = self.identity_loss(rec_photo, photo)
                    cycle_sketch_loss = self.identity_loss(rec_sketch, sketch)

                    # identity loss
                    idt_sketch = self.genAtoB(sketch)
                    idt_photo = self.genBtoA(photo)
                    idt_photo_loss = self.identity_loss(idt_photo, photo)
                    idt_sketch_loss = self.identity_loss(idt_sketch, sketch)

                    # total generator loss
                    gen_loss = (fake_sketch_loss + fake_photo_loss + (cycle_photo_loss + cycle_sketch_loss) * self.LAMBDA_CYCLE + (idt_photo_loss + idt_sketch_loss) * self.LAMBDA_IDENTITY)
                    
                self.opt_gen.zero_grad()
                g_scaler.scale(gen_loss).backward()                
                g_scaler.step(self.gen_disc)
                g_scaler.update()
                # print(fake_sketch_loss, fake_photo_loss, gen_loss)
                
                # print(f"Epoch: {epoch}, Iter: {idx}, Discriminator Loss: {disc_loss.item():.4f}, Generator Loss: {gen_loss.item():.4f}")
                
                # save images
                if idx % 5 == 0:
                    photo, fake_sketch, rec_photo  = photo*0.5+0.5, fake_sketch*0.5+0.5, rec_photo*0.5+0.5
                    sketch, fake_photo, rec_sketch  = sketch*0.5+0.5, fake_photo*0.5+0.5, rec_sketch*0.5+0.5
                    # img_train_list = [photo, fake_sketch, rec_photo, sketch,  fake_photo, rec_sketch]
                    sketch2photo_gen = [sketch, fake_photo, photo, rec_sketch]
                    photo2sketch_gen = [photo, fake_sketch, sketch, rec_photo]
                    save_train_examples(sketch2photo_gen, photo2sketch_gen, idx, self.folder_to_save_trained)
            
            gen_loss_arr.append(gen_loss.item())
            disc_loss_arr.append(disc_loss.item())
            
            # Checking for appending the loss
            if len(gen_loss_arr) > 0 and len(disc_loss_arr) > 0:
                print("fine!")
            else:
                print('problem here')
            
            # Print the log info
            if epoch % self.print_every == 0:
                print('Epoch [{:5d}/{:5d}] | disc_loss: {:6.4f} | gen_loss: {:6.4f}'.format(
                    epoch, self.EPOCHS + start_epoch, disc_loss.item(), gen_loss.item()))
            
            if self.SAVE_MODEL:
                CHECKPOINT_GEN_SKETCH = os.path.join(self.ROOT, f"epoch_{epoch}/cp/gen_sketch.pth.tar")
                CHECKPOINT_GEN_PHOTO = os.path.join(self.ROOT, f"epoch_{epoch}/cp/gen_photo.pth.tar")
                CHECKPOINT_DISC_SKETCH = os.path.join(self.ROOT, f"epoch_{epoch}/cp/disc_sketch.pth.tar")
                CHECKPOINT_DISC_PHOTO = os.path.join(self.ROOT, f"epoch_{epoch}/cp/disc_photo.pth.tar")

                save_checkpoint(self.genAtoB, self.opt_gen, filename=CHECKPOINT_GEN_SKETCH)
                save_checkpoint(self.genBtoA, self.opt_gen, filename=CHECKPOINT_GEN_PHOTO)
                save_checkpoint(self.discB, self.opt_disc, filename=CHECKPOINT_DISC_SKETCH)
                save_checkpoint(self.discA, self.opt_disc, filename=CHECKPOINT_DISC_PHOTO)
            
            with open(self.ROOT+"/gen_loss.json", 'w') as g_l:
                # indent=2 is not needed but makes the file human-readable
                json.dump(gen_loss_arr, g_l, indent=2)
                # print("done")
            # print("between!")
            with open(self.ROOT+"/disc_loss.json", 'w') as d_l:
                # indent=2 is not needed but makes the file human-readable
                json.dump(disc_loss_arr, d_l, indent=2)


    def test(self, dataloader, epoch, folder):
        self.genAtoB.eval()
        self.genBtoA.eval()

        for idx, (photo, sketch) in enumerate(dataloader):
            with torch.no_grad():
                photo = photo.to(self.device)
                sketch = sketch.to(self.device)
                fake_sketch = self.genAtoB(photo)
                fake_photo = self.genBtoA(fake_sketch)
                rec_photo = self.genBtoA(fake_sketch)
                rec_sketch = self.genAtoB(fake_photo)

                metrics(photo, sketch, fake_sketch, fake_photo, rec_photo, rec_sketch, self.ROOT, epoch)
                
                # save images
                if idx % 5 == 0:
                    photo, fake_sketch, fake_photo, rec_sketch, rec_photo = photo*0.5+0.5, fake_sketch*0.5+0.5, fake_photo*0.5+0.5, rec_sketch*0.5+0.5, rec_photo*0.5+0.5

                    photo_test_list = torch.cat([sketch, fake_photo, photo, rec_sketch], dim=3)
                    sketch_test_list = torch.cat([photo, fake_sketch, sketch, rec_photo], dim=3)

                    photo_gen_path = os.path.join(folder, f"photo_{idx}.png")
                    sketch_gen_path = os.path.join(folder, f"sketch_{idx}.png")

                    save_image(photo_test_list.data, photo_gen_path)
                    save_image(sketch_test_list.data, sketch_gen_path)
