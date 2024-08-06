import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    @TITLE
        Human Perception
    @RECOURCES:
        - (code) Lab 4
        - (theory) https://universe.bits-pilani.ac.in/uploads/JNKDUBAI/ImageProcessing7-FrequencyFiltering.pdf
        - (theory) https://www.mapleprimes.com/maplesoftblog/209594-Hybrid-Images-Visual-Perception-And-Distance-
        - (theory) https://en.wikipedia.org/wiki/Hybrid_image 
"""

def ideal_lpf(d0, n1, n2):
    # Use math.floor to ensure fit for even & uneven dimensions
    k1,k2 = np.meshgrid(np.arange(-round(n2/2)+1, math.floor(n2/2)+1), np.arange(-round(n1/2)+1, math.floor(n1/2)+1))
    d = np.sqrt(k1**2 + k2**2)
    h = np.zeros((n1,n2))
    h[d < d0] = 1
    return h
def ideal_hpf(d0,n1,n2):
    # Use math.floor to ensure fit for even & uneven dimensions
    k1,k2 = np.meshgrid(np.arange(-round(n2/2)+1, math.floor(n2/2)+1), np.arange(-round(n1/2)+1, math.floor(n1/2)+1))
    d = np.sqrt(k1**2 + k2**2)
    h = np.zeros((n1,n2))
    h[d > d0] = 1
    return h
def apply_filter(img,filter_mask):
    f = np.fft.fftshift(np.fft.fft2(img))
    f1 = f * filter_mask
    x1 = np.fft.ifft2(np.fft.ifftshift(f1))
    return x1

# ===================================
#   1.1 Apply filters images and
#       combine filtered images
#       in spatial and freq domain

# load images
img_1 = cv2.imread("images/face11.jpg",0)
img_2 = cv2.imread("images/face12.jpg",0)

# define our hpf and lpf transfer functions/masks
h1 = ideal_lpf(20,img_1.shape[0],img_1.shape[1])
h2 = ideal_hpf(20,img_2.shape[0],img_2.shape[1])

# bring images into frequency domain
f1 = np.fft.fftshift(np.fft.fft2(img_1))
f2 = np.fft.fftshift(np.fft.fft2(img_2))

# multiply with transfer funtions
f1_filtered = f1 * h1
f2_filtered = f2 * h2

# add in frequency domain (explain)
f_merged = f1_filtered + f2_filtered

# bring back into spatial domain using inverse fast fourier transform
img_fused_freqdomain = np.fft.ifft2(np.fft.ifftshift(f_merged))

# bring both individually into spatial
img_1_filtered = np.fft.ifft2(np.fft.ifftshift(f1_filtered))
img_2_filtered = np.fft.ifft2(np.fft.ifftshift(f2_filtered))

# add images in spatial domain
img_fused_spatial = abs(img_1_filtered)/255 + abs(img_2_filtered)/255

# original images
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(img_1,cmap="gray")
ax1.set_title("Image 1 (Spatial)")
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(img_2, cmap="gray")
ax2.set_title("Image 2 (Spatial)")
plt.show()

# filterd images spatial domain
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(abs(img_1_filtered)/255,cmap="gray")
ax1.set_title("Image 1 LPF (Spatial)")
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(abs(img_2_filtered)/255, cmap="gray")
ax2.set_title("Image 2 HPF (Spatial)")
plt.show()

# fused images
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(abs(img_fused_freqdomain)/255, cmap="gray")
ax1.set_title("Hybrid Image (Frequency Domain)")
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(img_fused_spatial, cmap="gray")
ax2.set_title("Hybrid Image (Spatial Domain)")
plt.show()

