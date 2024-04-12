import numpy as np
from tqdm import tqdm

# FISTA reconstruction functions
def gradientImage(I):
  Ix = np.zeros(I.shape)
  Iy = np.zeros(I.shape)
  Ix[:, :-1] = I[:, 1:] - I[:, :-1]
  Iy[:-1, :] = I[1:, :] - I[:-1, :]
  return (Ix, Iy)

def divergenceImage(vx, vy):
  dv = np.zeros(vx.shape)
  dv[:, 1:] = dv[:, 1:] + vx[:, 1:] - vx[:, :-1]
  dv[1:, :] = dv[1:, :] + vy[1:, :] - vy[:-1, :]
  return dv

def totalVariationImage(I):
  (Ix, Iy) = gradientImage(I)
  TV = np.sum(np.sqrt(Ix**2 + Iy**2), axis=None)
  return TV

def proximalTV(I, Lambda, Iter):
# We find vx and vy by minimizing ||I - lambda div(vx, vy)||2 under the
# condition of ||(vx, vy)|| <= 1
# We can minimize this by gradient descent
# Then the new image is just I - lambda div(vx, vy) and its total
# variation can be proved to decrease after this step because this new I is
# the optimization point of ||I_new - I||2 + 2 lambda TV(I_new)
#
# For more details about the algorithm, please refer to this paper: Beck,
# Amir, and Marc Teboulle. "Fast gradient-based algorithms for constrained
# total variation image denoising and deblurring problems." IEEE
# transactions on image processing 18.11 (2009): 2419-2434.
  lr = 1/(16 * Lambda)
  L = np.zeros(Iter)
  (I_x, I_y) = gradientImage(I)
  vx = np.zeros(I.shape)
  vy = np.zeros(I.shape)
  # For the acceleration step
  acvx = np.zeros(I.shape)
  acvy = np.zeros(I.shape)

  T = 1
  for k in range(0, Iter):
    T_pre = T
    T = (1 + np.sqrt(1 + 4 * T**2)) / 2
    vx_pre = vx
    vy_pre = vy

    dv = divergenceImage(acvx, acvy)
    # The derivative for gradient descent
    I_p = I - Lambda * dv
    I_p[I_p < 0] = 0 # Nonnegativity constraint
    (G_x, G_y) = gradientImage(I_p)
    G_x = 2 * G_x
    G_y = 2 * G_y

    # L is the thing we want to minimize
    L[k] = np.sum(I_p ** 2, axis=None)

    # Gradient descent
    vx = acvx - lr * G_x
    vy = acvy - lr * G_y

    vm = np.sqrt(vx**2 + vy**2)
    pr = vm;
    pr[vm <= 1] = 1
    vx = vx / pr
    vy = vy / pr

    # FISTA acceleration
    acvx = (T_pre - 1) / T * (vx - vx_pre) + vx
    acvy = (T_pre - 1) / T * (vy - vy_pre) + vy

  dv = divergenceImage(vx, vy)
  I_p = I - Lambda * dv
  I_p[I_p < 0] = 0 # Nonnegativity constraint


  TV = totalVariationImage(I)
  TV_p = totalVariationImage(I_p)
  Dis = np.sum((I_p - I) ** 2, axis=None)
  # print("----Dual problem objective: before is %f" %(L[0]))
  # print("----Dual problem objective: middle is %f" %(L[round(Iter/2)]))
  # print("----Dual problem objective: after is %f" %(L[-1]))
  # print("----Before proximal, ||I - I||2 + 2 lambda TV(I) is %f" %(2 * Lambda * TV))
  # print("----After proximal, ||I_p - I||2 + 2 lambda TV(I_p) is %f" %(Dis + 2 * Lambda * TV_p))

  return I_p

def reconImageFISTA(A, b, M, N, Lambda, L, Iter, Iter_sub):

  T = 1
  Loss = np.zeros(Iter)
  x = np.zeros(M * N)
  y = np.zeros(M * N)
  for k in tqdm(range(0, Iter)):
    # print("==========FISTA iteration number %d / %d==========" %(k, Iter))
    T_pre = T
    x_pre = x
    # Compute x = x - 2 / L * AT(Ax - b)
    Res = np.sum((np.dot(A, y) - b) ** 2, axis=None)
    # print("Before gradient step the Residule is %f" %(Res))
    y = y - (2 / L) * np.transpose(A) @ (A @ y - b)
    Res = np.sum((np.dot(A, y) - b) ** 2, axis=None)
    # print("After gradient step the Residule is %f" %(Res))

    # Compute x = TVdenoise(x)
    I = y.reshape(N, M)
    TV = totalVariationImage(I)
    # print("Before TV denoising TV is %f" %(TV))
    I = proximalTV(I, Lambda * 2 / L, Iter_sub)
    TV = totalVariationImage(I)
    # print("After TV denoising TV is %f" %(TV))
    x = I.reshape(N * M)

    # Do FISTA acceleration
    T = (1 + np.sqrt(1 + 4 * T**2)) / 2
    y = x + (T_pre - 1) / T * (x - x_pre)

    I = x.reshape(N, M)
    Res = np.sum((np.dot(A, x) - b) ** 2, axis=None)
    TV = totalVariationImage(I)
    Loss[k] = Res + 2 * Lambda * TV
    # print("The loss function is %f, Residule is %f, TV is %f" %(Loss[k], Res, TV))
  return (I, Loss)


def fista_baseline(signals, A, M, N, Lambda=0.001, Iter=100, Iter_sub=1000, Anorm=0.02):
  num_imgs = signals.shape[0]
  L = 2 * Anorm * Anorm
  signals_flat = signals.reshape(num_imgs, -1)

  fista_recons = np.zeros((num_imgs, M, N))
  for i in range(num_imgs):
    print("Recon " + str(i + 1) + ": ", sep='')
    I, Loss = reconImageFISTA(np.transpose(A.T.numpy()).astype(float),
                              signals_flat[i],
                              M, N, Lambda, L, Iter, Iter_sub)
    fista_recons[i] = I

  return fista_recons










