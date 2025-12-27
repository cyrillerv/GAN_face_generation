import torch
import os
import torch.onnx
import imageio.v2 as imageio
import re
import glob
from src.models import weights_init

def load_model(checkpoint_path, netG, netD, optimizerD, optimizerG, reset=False) :
    if os.path.exists(checkpoint_path) and not reset :
        print(f"Chargement du checkpoint : {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        G_losses = checkpoint['G_losses']
        D_losses = checkpoint['D_losses']
        print(f"Reprise de l'entraînement à l'époque {start_epoch}")
    else:
        print("Aucun checkpoint trouvé. Démarrage d'un nouvel entraînement.")
        folder_path = './generated_faces'
        files = glob.glob(os.path.join(folder_path, '*.png'))
        for f in files:
            os.remove(f)

        netG.apply(weights_init)
        netD.apply(weights_init)
        start_epoch = 0
        G_losses = []
        D_losses = []

    return netG, netD, optimizerG, optimizerD, start_epoch, G_losses, D_losses


def save_model_prod(Generator, conf, checkpoint_path, output_pth, output_onnx) :
    """Sauvegarde le modèlke au dernier checkpoint sauvegardé"""
    netG = Generator(conf.ngpu, conf.nz, conf.ngf, conf.nc).to(conf.device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netG.eval()
        
        torch.save(netG.state_dict(), output_pth)
        print(f"Modèle PyTorch léger sauvegardé : {output_pth}")

        dummy_input = torch.randn(1, 100, 1, 1, device=conf.device)
        
        torch.onnx.export(netG, 
                        dummy_input, 
                        output_onnx, 
                        verbose=False, 
                        input_names=['input_noise'], 
                        output_names=['output_image'], 
                        dynamic_axes={'input_noise': {0: 'batch_size'}, 
                                        'output_image': {0: 'batch_size'}})
        print(f"Modèle ONNX exporté : {output_onnx}")

    else:
        print("Erreur : Aucun checkpoint trouvé à exporter.")



def animate_gif(image_folder, gif_name, pattern) :
    files = [f for f in os.listdir(image_folder) if f.endswith('.png') and pattern in f]

    files.sort(key=lambda f: int(re.sub(r'\D', '', f)))

    images = []
    for filename in files:
        file_path = os.path.join(image_folder, filename)
        images.append(imageio.imread(file_path))

    imageio.mimsave(gif_name, images, duration=500, loop=0)
    print(f"GIF sauvegardé sous : {gif_name}")