# Diffusion Score-Based Model for Photoacoustic Tomography (PAT)

This repo contains the code for the paper "SCORE-BASED DIFFUSION MODELS FOR PHOTOACOUSTIC TOMOGRAPHY IMAGE RECONSTRUCTION," presented at ICASSP 2024.

Authors: Sreemanti Dey, Snigdha Saha, Berthy T. Feng, Manxiu Cui, Laure Delisle, Oscar Leong, Lihong V. Wang, Katherine L. Bouman.

--------------------

Abstract: Photoacoustic tomography (PAT) is a rapidly-evolving medical imaging modality that combines optical absorption contrast with ultrasound imaging depth. One challenge in PAT is image reconstruction with inadequate acoustic signals due to limited sensor coverage or due to the density of the transducer array. Such cases call for solving an ill-posed inverse reconstruction problem. In this work, we use score-based diffusion models to solve the inverse problem of reconstructing an image from limited PAT measurements. The proposed approach allows us to incorporate an expressive prior learned by a diffusion model on simulated vessel structures while still being robust to varying transducer sparsity conditions.

We provide a showcase of our work and results in [Demo_Notebook.ipynb](https://github.com/sreemanti-dey/diffusion_for_PAT/blob/main/Demo_Notebook.ipynb). We suggest running the .ipynb in Google Colab and selecting runtime to be GPU, but this can also be run locally.

--------------------

The folders 'models' and 'op', as well as the files 'datasets.py', 'evaluation.py', 'likelihood.py', and 'losses.py' are from Song et al's [code base](https://github.com/yang-song/score_sde_pytorch) from the paper [Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole, “Score-based generative modeling through stochastic differential equations,” 2021](https://openreview.net/forum?id=PxTIG12RRHS). 


@inproceedings{
  song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=PxTIG12RRHS}
}

