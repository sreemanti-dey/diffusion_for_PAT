import torch
import random
from pat import forwardMatrixFullRingCDMMI
from misc_utils import awgn

# noisy dataset
class AddGaussianNoise():
    ''' Adds some Gaussian Noise ~ N(0, std^2 I) to an image
    '''
    # structure for custom transform follows pytorch source code
    # https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html
    def __init__(self, std=1.):
        self.std = std

    def __call__(self, image):
        ''' Add noise, returns noisy image'''
        noise = self.std * torch.randn_like(image)
        return image + noise
    def __repr__(self):
        return f"{self.__class__.__name__}()"

# Set up inpainting
def inpaint(images, ratio=0.05):
    num_pixels = images.shape[-2] * images.shape[-1]
    num_samples = int(ratio * num_pixels)
    # create subsampling matrix L
    L = torch.eye(num_pixels, device=images.device)
    for pixel in random.sample(range(0, num_pixels), num_samples):
        L[pixel][pixel] = 0
    # black out pixels in images using L (a binary matrix with zeroes where we want to black out pixels)
    inpainted_images = images.clone()
    for i in range(len(images)):
        inpainted_images[i] = torch.reshape(torch.matmul(L, inpainted_images[i].view(num_pixels)), images[0].shape)
    return inpainted_images, L

# Set up Gaussian downsampling
def gaussian_downsampling(images, keep_ratio=0.95):
    num_pixels = images.shape[-2] * images.shape[-1]
    num_samples = int(keep_ratio * num_pixels) # how many pixels do we KEEP

    # create well-conditioned gaussian T
    T = torch.normal(0, 1, size=(num_pixels, num_pixels), device=images.device)
    U, S, V = torch.linalg.svd(T)
    T = torch.matmul(U, torch.t(V))

    # create subsampling matrix L
    L = torch.zeros((num_pixels, num_pixels), device=images.device)
    pixels_to_keep = torch.tensor(random.sample(range(0, num_pixels), num_samples), device=images.device)
    for pixel in pixels_to_keep:
        L[pixel][pixel] = 1
    
    # define downsampling operator P
    P = torch.index_select(torch.eye(num_pixels, device=images.device), 0, pixels_to_keep)

    # apply y = P(L)Tx
    A = torch.matmul(torch.matmul(P, L), T)
    gauss_down_images = torch.zeros(images.shape[0], A.shape[0], 1, device=images.device)
    for i in range(len(images)):
        gauss_down_images[i] = torch.matmul(A, images[i].reshape((-1, 1)))
    gauss_down_images = gauss_down_images.reshape(images.shape[0], 1, -1, images.shape[-1])

    return gauss_down_images, P, L, T

# Set up PAT
def PAT_forward(images, PAT_config, forward_A=None, add_noise=False, noise=0.0, remove_transducers=False, removed_transducers=None):
    if forward_A == None:
      A, _, _, _ = forwardMatrixFullRingCDMMI(*PAT_config)
      # the data acquisition matrix T is A^t
      T = torch.tensor(A.T, device=images.device).float()

      if remove_transducers:
          # define a reduced T matrix 
          pixels_to_keep = list(range(0, T.shape[0]))
          for td in removed_transducers:
              N_sample = PAT_config[7]
              for pixel in range(td * N_sample, (td + 1)* N_sample):
                  pixels_to_keep.remove(pixel)
          pixels_to_keep = torch.tensor(pixels_to_keep, device=images.device)
          T_red = torch.index_select(T, 0, pixels_to_keep)
          T = T_red

      P, L = [torch.eye(T.shape[0], device=images.device)] * 2

      # apply y = P(L)Tx
      A = torch.matmul(torch.matmul(P, L), T)
    else:
      A = forward_A

    PAT_images = torch.zeros(images.shape[0], A.shape[0], 1, device=images.device)
    for i in range(len(images)):
        PAT_images[i] = torch.matmul(A, images[i].reshape((-1, 1)))
        if add_noise:
          # PAT_images[i] = PAT_images[i] + noise * torch.randn_like(PAT_images[i])
          PAT_images[i] = torch.tensor(awgn(PAT_images[i].numpy()))
    if remove_transducers:
        N_transducer = PAT_config[0]
        N_sample = PAT_config[7]
        PAT_images = PAT_images.reshape(images.shape[0], 1, N_transducer - len(removed_transducers), N_sample)
    else:
        N_transducer = PAT_config[0]
        PAT_images = PAT_images.reshape(images.shape[0], 1, N_transducer, -1)

    if forward_A == None:
      return PAT_images, P, L, T
    return PAT_images
