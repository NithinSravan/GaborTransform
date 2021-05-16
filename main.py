from skimage.filters import gabor
from skimage import data, io
from matplotlib import pyplot as plt
from phasepack import phasecong,phasecongmono,phasesym,phasesymmono
import numpy as np
import math
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
import cv2
from gabor import compute_feats,mag_phase,create_kernels
import os

import sys
import glob
import concurrent.futures


# prepare filter bank kernels
kernels = create_kernels()
image = io.imread('vlcsnap-2019-06-21-10h51m33s614.png')
image_r = image[:, :, 0]
filt_real,filt_imag=compute_feats(image_r, kernels)

plt.figure()

io.imshow(filt_imag)
io.show()

pcm = phasecong(filt_imag)

y = pcm[0]
io.imshow(y)
io.show()


mser = cv2.MSER_create()

x= np.array(y, dtype=np.uint8)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if x[i][j] > 0:
            x[i][j]=255
        else:
            x[i][j]=0
color = np.ndarray((x.shape[0], x.shape[1], 3), dtype=np.uint8)
color[:,:,0] = x
color[:,:,1] = x
color[:,:,2] = x

img = x.copy()

# detect regions in gray scale image
regions, _ = mser.detectRegions(color)
print(regions)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

cv2.polylines(image, hulls, 1, (0, 255, 0))

cv2.imshow('image',image)

cv2.waitKey(0)

"""mask = np.zeros((color.shape[0], color.shape[1], 1), dtype=np.uint8)

for contour in hulls:
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

#this is used to find only text regions, remaining are ignored
text_only = cv2.bitwise_and(color, color, mask=mask)

cv2.imshow("text only", text_only)

cv2.waitKey(0)"""

"""
image_r = io.imread('vlcsnap-2019-06-21-10h51m33s614.png')[:,:,2]
image_g = io.imread('vlcsnap-2019-06-21-10h51m33s614.png')[:,:,1]
image_b = io.imread('vlcsnap-2019-06-21-10h51m33s614.png')[:,:,0]


# detecting edges
filt_real_r, filt_imag_r = gabor(image_r, frequency=0.6)
filt_real_g, filt_imag_g = gabor(image_g, frequency=0.6)
filt_real_b, filt_imag_b = gabor(image_b, frequency=0.6)
plt.figure()



red_pcm = phasecong(filt_imag_r)
green_pcm = phasecong(filt_imag_g)
blue_pcm = phasecong(filt_imag_b)

io.imshow(an[0])
io.show()

for i in range(image_r.shape[0]):
    for j in range(image_r.shape[1]):
        i_plus= i + math.sin()
mser = cv2.MSER_create()

#np.array(an[0], dtype=np.uint8)

x= filt_imag
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if x[i][j] > 0:
            x[i][j]=255
        else:
            x[i][j]=0
color = np.ndarray((x.shape[0], x.shape[1], 3), dtype=np.uint8)
color[:,:,0] = x
color[:,:,1] = x
color[:,:,2] = x

img = x.copy()

# detect regions in gray scale image
regions, _ = mser.detectRegions(color)
print(regions)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

cv2.polylines(color, hulls, 1, (0, 255, 0))

cv2.imshow('img',color)

cv2.waitKey(0)

mask = np.zeros((color.shape[0], color.shape[1], 1), dtype=np.uint8)

for contour in hulls:
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

#this is used to find only text regions, remaining are ignored
text_only = cv2.bitwise_and(color, color, mask=mask)

cv2.imshow("text only", text_only)

cv2.waitKey(0)
if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train = True
    else:
        train = False

    path = sys.argv[2]
    image_list = glob.glob(os.path.join(path, '*.jpg'))

    verbose = len(sys.argv) > 3

    print('running in {} mode'.format(sys.argv[1]))
    print('found {} images'.format(len(image_list)))

    if verbose:
        print(image_list)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for image_name in image_list:
            futures.append(executor.submit(process_file, image_name=image_name, verbose=verbose))
        for future in concurrent.futures.as_completed(futures):
            print(future.result())"""