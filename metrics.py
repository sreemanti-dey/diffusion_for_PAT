import numpy as np
import torch
from torch.serialization import normalize_storage_type 
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from statistics import mean, stdev


# normalize to [0,1]
def scale(x):
  return (x-np.min(x))/(np.max(x)-np.min(x))

# lower --> more similar images
def pixel_to_pixel_MSE(original, noisy): 
    return np.mean((original - noisy)**2)

# higher --> more similar images
def PSNR(original, noisy, data_range=1):
    if not isinstance(original, np.ndarray):
      original = original.cpu().numpy()
    if not isinstance(noisy, np.ndarray):
      noisy = noisy.cpu().numpy()
    return psnr(original, noisy, data_range=data_range)

def avg_PSNR(GT, preds):
  # GT and preds must have the same shape
  psnrs = []
  for i in range(len(GT)):
    psnrs.append(PSNR(GT[i], preds[i]))
  return mean(psnrs), stdev(psnrs)

def compare_PSNR(GT_all, fista_all, sau_all, diff_all):
  counter = 0
  for i in range(GT_all.shape[0]):
    GT = GT_all[i]
    fista = PSNR(GT, fista_all[i])
    sau = PSNR(GT, sau_all[i])
    diff = PSNR(GT, diff_all[i])

    if diff > fista and diff > sau:
      counter += 1
  
  return (counter / GT_all.shape[0]) * 100

# higher --> more similar images
def SSIM(original, noisy):
    if not isinstance(original, np.ndarray):
      original = original.cpu().numpy()
    if not isinstance(noisy, np.ndarray):
      noisy = noisy.cpu().numpy()
    return ssim(original, noisy)

def avg_SSIM(GT, preds):
  # GT and preds must have the same shape
  ssims = []
  for i in range(len(GT)):
    ssims.append(SSIM(GT[i], preds[i]))
  return mean(ssims), stdev(ssims)


def LPIPS(original, noisy, net='alex'):
    # expecting original and noisy to be NxHxW
    if not torch.is_tensor(original):
      original = torch.tensor(scale(original))
    if not torch.is_tensor(noisy):
      noisy = torch.tensor(scale(noisy))

    original = torch.tile(torch.unsqueeze(original, 1), (1,3,1,1))
    noisy = torch.tile(torch.unsqueeze(noisy, 1), (1,3,1,1))
    lpips = LearnedPerceptualImagePatchSimilarity(net_type=net, normalize=True)
    return lpips(original, noisy).item()
    
