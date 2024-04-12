# utility functions

import torch
import numpy as np
import matplotlib.pyplot as plt 
from metrics import scale, PSNR
from torchvision.utils import make_grid


# makes a square grid of images
def plot_samples(images):
    if torch.is_tensor(images):
      images = images.squeeze().cpu().detach().numpy()

    nrow = int(np.sqrt(images.shape[0]))
    fig, axs = plt.subplots(nrow, nrow, figsize=(8, 8))

    for i in range(images.shape[0]):
      axs.flatten()[i].imshow(images[i], cmap='gray')
      axs.flatten()[i].axis('off')

    plt.show()


# plots images in two rows, top row before and bottom row after
def plot_before_after(clean_images, imgs_before, imgs_after, title=""):
    assert(imgs_before.shape[0] == imgs_after.shape[0])
    fig, axs = plt.subplots(2, imgs_before.shape[0], figsize=(16, 5))
    for i, image in enumerate(imgs_before):
        im = axs[0][i].imshow(image.cpu().permute(1, 2, 0).squeeze(), cmap='gray')
        axs[0][i].set_xticks([])
        axs[0][i].set_yticks([])
    for i, image in enumerate(imgs_after):
        im = axs[1][i].imshow(image.cpu().permute(1, 2, 0).squeeze(), cmap='gray')
        axs[1][i].set_xticks([])
        axs[1][i].set_yticks([])
        plt.colorbar(im, ax=axs[1,i])
        if clean_images is not None:
            clean = clean_images[i].cpu().permute(1,2,0).squeeze()
            noisy = image.cpu().permute(1,2,0).squeeze()
            psnr_val = PSNR(clean, noisy).item()
            axs[1][i].set_title('PSNR: {:.3f}'.format(psnr_val), y=-0.2)
    fig.suptitle(title, size=20)


# Adding white gaussian noise
# snr in dB
def awgn(x, snr=30):
  signal_power = np.mean(x ** 2)
  noise_power = signal_power / (10 ** (snr / 10.0))
  noise = np.random.normal(scale=np.sqrt(noise_power), size=x.shape)
  x_with_noise = x + noise
  return x_with_noise


# Given the number of transducers, returns the ones remaining
# as a list between 0 and N_transducer
def limited_view_rmd(N_transducer, N_keep):
  # the parity of N_transducer and N_keep must match
  edge = (N_transducer - N_keep) / 2
  return [i for i in range(N_transducer) if i < edge or i >= (N_transducer - edge)]


# Analogous for spatial aliasing setting
def spatial_alias_rmd(N_transducer, skip):
  return [i for i in range(0, N_transducer) if i % skip != 0]
