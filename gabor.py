import numpy as np
import math
from skimage.filters import gabor
from skimage import data, io
from matplotlib import pyplot as plt


from skimage.filters import gabor_kernel,gabor
from scipy import ndimage as ndi
import cv2

# global filt_real, filt_imag
""" filt_real=filt_imag=image
 for k, kernel in enumerate(kernels):
     filt_real = ndi.convolve(filt_real, np.real(kernel), mode='wrap')
     filt_imag = ndi.convolve(filt_imag, np.imag(kernel), mode='wrap')
 return filt_real,filt_imag"""
def compute_feats(image, kernels):
    global filt_real, filt_imag
    filt_imag=filt_real=image
    mean=np.zeros(image.shape,dtype=np.uint8)
    for i in range(8):
        """ filt_real = ndi.convolve(filt_real, np.real(kernel), mode='wrap')
         filt_imag = ndi.convolve(filt_real, np.imag(kernel),mode='wrap')"""
        f=0.1+i/10
        print(f)
        filt_real,filt_imag=gabor(image,frequency=f)
        mean+=filt_imag

    return filt_real,filt_imag

def mag_phase():
    mag = np.ndarray((filt_imag.shape[0], filt_imag.shape[1]), dtype=np.uint8)
    phase = np.ndarray((filt_imag.shape[0], filt_imag.shape[1]), dtype=np.uint8)
    for i in range(len(filt_imag)):
        for j in range(len(filt_imag[i])):
            mag[i][j] = math.sqrt(filt_real[i][j] ** 2 + filt_imag[i][j] ** 2)
            if filt_real[i][j] != 0:
                tan = filt_imag[i][j] / filt_real[i][j]
            else:
                tan = np.inf

            phase[i][j] = math.atan(tan)
    return mag,phase

# prepare filter bank kernels
def create_kernels():
    kernels = []
    for frequency in (0.05, 0.7):
            kernel = gabor_kernel(frequency)
            kernels.append(kernel)
    return kernels








"""
# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma)
            kernels.append(kernel)


"""