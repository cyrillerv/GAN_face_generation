import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from config import Config
from src.models import *
from src.data import *
from src.load_and_save import *
from src.plot_graphs import *



if __name__ == "__main__":
    conf = Config()
    download_data(conf.dataroot)

    dataloader = load_data(conf.dataroot, conf.dataset_size, conf.image_size, conf.batch_size, conf.workers)

    netG = Generator(conf.ngpu, conf.nz, conf.ngf, conf.nc).to(conf.device)
    netD = Discriminator(conf.ngpu, conf.ndf, conf.nc).to(conf.device)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, conf.nz, 1, 1, device=conf.device)
    real_label = 0.9
    fake_label = 0.1

    # Pour le dev
    checkpoint_path = 'last_checkpoint.pth'
    # Pour la prod
    output_pth = r'GAN_prod\generator_final.pth'
    output_onnx = r'GAN_prod\generator.onnx'

    img_list = []

    optimizerD = optim.Adam(netD.parameters(), lr=conf.lr_D, betas=(conf.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=conf.lr_G, betas=(conf.beta1, 0.999))
    
    netG, netD, optimizerG, optimizerD, start_epoch, G_losses, D_losses = load_model(checkpoint_path, netG, netD, optimizerD, optimizerG)

    print("Starting Training Loop...")

    for epoch in range(start_epoch, conf.num_epochs):
        for i, data in enumerate(dataloader, 0):
            
            netD.zero_grad()
            real_cpu = data[0].to(conf.device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=conf.device)

            # We add some noise to the real images so the discriminator does not learn to quick
            noise_factor = max(0, 1.0 - (epoch / (conf.num_epochs / 2)))
            real_cpu_noisy = real_cpu + (0.1 * noise_factor * torch.randn_like(real_cpu))

            output = netD(real_cpu_noisy).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, conf.nz, 1, 1, device=conf.device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label) 
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, conf.num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(errG.item())
            D_losses.append(errD.item())

        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        vutils.save_image(fake, f'generated_faces/resultat_epoque_{epoch}.png', normalize=True)

        torch.save({
            'epoch': epoch,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'G_losses': G_losses,
            'D_losses': D_losses
        }, checkpoint_path)
        print(f"Checkpoint sauvegardé pour l'époque {epoch}")

    plot_losses(G_losses, D_losses)
    save_model_prod(Generator, conf, checkpoint_path, output_pth, output_onnx)
    image_folder = 'generated_faces'
    gif_name = 'evolution_entrainement.gif'
    pattern = 'resultat_epoque'

    animate_gif(image_folder, gif_name, pattern)

