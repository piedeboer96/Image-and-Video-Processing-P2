import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import imageio as iio


from numpy import pi
from numpy import r_

"""
    @TITLE:
        DCT Domain Watermark Inserttion
    @RECOURCES:
        - (code) lab 7 
        - (theory) gonzalez - example 8.30 page 627
        - (theory) https://www.sciencedirect.com/topics/computer-science/frequency-coefficient
        - (implementation) https://stackoverflow.com/questions/15978468/using-the-scipy-dct-function-to-create-a-2d-dct-ii
        - (images) https://people.math.sc.edu/Burkardt/data/tif/tif.html
        - (images) https://eeweb.engineering.nyu.edu/~yao/EL5123/SampleData.html
        - (theory) https://dsp.stackexchange.com/questions/17224/why-zig-zag-manner-scan-is-used-in-dct-for-image-compression
        - (code) https://github.com/getsanjeev/compression-DCT/blob/master/zigzag.py
        - (theory) https://web.stanford.edu/class/ee368b/Handouts/11-TransformCoding.pdf
        - (code) https://nl.mathworks.com/matlabcentral/answers/336918-how-to-calculate-and-plot-the-histogram-of-dct-coefficients-of-an-image?s_tid=prof_contriblnk
        - (theory) https://www.researchgate.net/figure/The-original-cover-image-a-watermarked-image-b-histogram-of-the-original-cover_fig3_346598857    
    @NOTE:
        - I ADAPTED THE MATLAB AND PYTHON CODE FROM THE LAB
        - THE INVERSE ZIGSCAN IS DIRECTLY FROM GITHUB... 
"""

"""
    Methods
"""


def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a.T, norm='ortho' ).T, norm='ortho' )
def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.T , norm='ortho').T,norm='ortho')
def zigzag(input): 
# initializing the variables

    h = 0 
    v = 0 
    vmin = 0
    hmin = 0
    vmax = input.shape[0]
    hmax = input.shape[1]
    i = 0
    output = np.zeros((vmax * hmax))
    
    while ((v < vmax) and (h < hmax)):

        if ((h + v) % 2) == 0:
            if (v == vmin):
                output[i] = input[v,h]
                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1
                i = i + 1
            else:
                if ((h == hmax-1) and (v < vmax)):
                    output[i] = input[v,h]
                    v = v + 1
                    i = i + 1
                else:
                    if ((v > vmin) and (h < hmax-1)):
                        output[i] = input[v,h]
                        v = v - 1
                        h = h + 1
                        i = i + 1
        else:
            if ((v == vmax-1) and (h <= hmax-1)):
                output[i] = input[v,h]
                h = h + 1
                i = i + 1
            else:
                if (h == hmin):
                    output[i] = input[v,h]
                    if (v == vmax-1):
                        h = h + 1
                    else:
                        v = v + 1
                    i = i + 1
                else:
                    if ((v < vmax-1) and (h > hmin)):
                        output[i] = input[v,h]
                        v = v + 1
                        h = h - 1
                        i = i + 1
        if ((v == vmax-1) and (h == hmax-1)):
            output[i] = input[v,h]
            break

    
    return output
def inverse_zigzag(input,vmax,hmax):
	
	#print input.shape

	# initializing the variables
	#----------------------------------
	h = 0
	v = 0

	vmin = 0
	hmin = 0

	output = np.zeros((vmax, hmax))

	i = 0
    #----------------------------------

	while ((v < vmax) and (h < hmax)): 
		#print ('v:',v,', h:',h,', i:',i)   	
		if ((h + v) % 2) == 0:                 # going up
            
			if (v == vmin):
				#print(1)
				
				output[v, h] = input[i]        # if we got to the first line

				if (h == hmax):
					v = v + 1
				else:
					h = h + 1                        

				i = i + 1

			elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
				#print(2)
				output[v, h] = input[i] 
				v = v + 1
				i = i + 1

			elif ((v > vmin) and (h < hmax -1 )):    # all other cases
				#print(3)
				output[v, h] = input[i] 
				v = v - 1
				h = h + 1
				i = i + 1

        
		else:                                    # going down

			if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
				#print(4)
				output[v, h] = input[i] 
				h = h + 1
				i = i + 1
        
			elif (h == hmin):                  # if we got to the first column
				#print(5)
				output[v, h] = input[i] 
				if (v == vmax -1):
					h = h + 1
				else:
					v = v + 1
				i = i + 1
        		        		
			elif((v < vmax -1) and (h > hmin)):     # all other cases
				output[v, h] = input[i] 
				v = v + 1
				h = h - 1
				i = i + 1




		if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
			#print(7)        	
			output[v, h] = input[i] 
			break


	return output
def uppertriangularmask(input):

    # k=3 , DC and AC
    # mask = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
    #                  [1, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0]])

    # k=9 , DC and AC
    # mask = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
    #                  [1, 1, 1, 0, 0, 0, 0, 0],
    #                  [1, 1, 0, 0, 0, 0, 0, 0],
    #                  [1, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0]])

    # k=21 , DC and AC
    # mask = np.array([[1, 1, 1, 1, 1, 1, 0, 0],
    #                  [1, 1, 1, 1, 1, 0, 0, 0],
    #                  [1, 1, 1, 1, 0, 0, 0, 0],
    #                  [1, 1, 1, 0, 0, 0, 0, 0],
    #                  [1, 1, 0, 0, 0, 0, 0, 0],
    #                  [1, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0]])
    
    # k=28 , DC and AC
    # mask = np.array([[1, 1, 1, 1, 1, 1, 1, 0],
    #                  [1, 1, 1, 1, 1, 1, 0, 0],
    #                  [1, 1, 1, 1, 1, 0, 0, 0],
    #                  [1, 1, 1, 1, 0, 0, 0, 0],
    #                  [1, 1, 1, 0, 0, 0, 0, 0],
    #                  [1, 1, 0, 0, 0, 0, 0, 0],
    #                  [1, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0]])
    
    # k=28-1 , only AC
    mask = np.array([[0, 1, 1, 1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 1, 0, 0, 0],
                     [1, 1, 1, 1, 0, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0]])


    return mask.flatten() * input.flatten()
def getuppertriangularmask():
    
    # k=3 , DC and AC
    # mask = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
    #                  [1, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0]])

    # k=9 , DC and AC
    # mask = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
    #                  [1, 1, 1, 0, 0, 0, 0, 0],
    #                  [1, 1, 0, 0, 0, 0, 0, 0],
    #                  [1, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0]])

    # k=21 , DC and AC
    # mask = np.array([[1, 1, 1, 1, 1, 1, 0, 0],
    #                  [1, 1, 1, 1, 1, 0, 0, 0],
    #                  [1, 1, 1, 1, 0, 0, 0, 0],
    #                  [1, 1, 1, 0, 0, 0, 0, 0],
    #                  [1, 1, 0, 0, 0, 0, 0, 0],
    #                  [1, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0]])

    # k=28 , DC and AC
    # mask = np.array([[1, 1, 1, 1, 1, 1, 1, 0],
    #                  [1, 1, 1, 1, 1, 1, 0, 0],
    #                  [1, 1, 1, 1, 1, 0, 0, 0],
    #                  [1, 1, 1, 1, 0, 0, 0, 0],
    #                  [1, 1, 1, 0, 0, 0, 0, 0],
    #                  [1, 1, 0, 0, 0, 0, 0, 0],
    #                  [1, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0]])
    
    # k=28-1 , only AC
    mask = np.array([[0, 1, 1, 1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 1, 0, 0, 0],
                     [1, 1, 1, 1, 0, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0]])



    return mask  

"""
    Main Code
"""

# ==================================================== #
#    2.1 Compute 2D DCT of the image to be watermarked

img = iio.imread("images/lena512.bmp").astype(float)
imsize = img.shape
dct = np.zeros(imsize)
block_size = 8  
for i in r_[:imsize[0]:block_size]:
    for j in r_[:imsize[1]:block_size]:
        dct[i:i+block_size,j:j+block_size] = dct2( img[i:i+block_size,j:j+block_size] )

# extract spatial block
# dct block from image
# pos=128
# fig = plt.figure(figsize=(10,8))
# ax1 = fig.add_subplot(1,2,1)
# ax1.imshow(img[pos:pos+block_size,pos:pos+block_size],cmap='gray')
# ax1.set_title("An 8x8 sptial block")
# ax2 = fig.add_subplot(1,2,2)
# ax2.imshow(dct[pos:pos+block_size,pos:pos+block_size],cmap='gray',vmax= np.max(dct)*0.01,vmin = 0, extent=[0,pi,pi,0])
# ax2.set_title( "An 8x8 DCT block")
# plt.show()


# NOTE: this just showed a block... now display the full DCT image
# plt.figure()
# plt.imshow(dct,vmax = np.max(dct)*0.01,vmin = 0)
# plt.title( "8x8 DCTs of the image")

# ==================================================== #
#   2.2 Locate K largest coefficients using zigzag scan

# we need to iterate with steps of size 8 (blocksize)
img2 = img
height  = len(img2[1])
width   = len(img2[0])   
im = img2
dct_domain        = np.zeros_like(img2)
dct_k_coeffs      = np.zeros((64,len(np.arange(1,width,8)) * len(np.arange(1,height,8)))) 
dct_reconstrcuted = np.zeros_like(img2)

# build arrays to reconstruct the image
img_reconstructed_kcoefs    = np.zeros_like(img2)
img_reconstructed_allcoefs  = np.zeros_like(img2)

# sliding window
for i1 in np.arange(0,width,8).reshape(-1):
    for i2 in np.arange(0,height,8).reshape(-1):
        zBLOCK = im[i1:i1+8, i2:i2 +8]
      
# Forward Discrete Cosine Transform
        win1 = dct2(zBLOCK)
        dct_domain[i1:i1+8, i2:i2 +8] = win1

# Keep the k best coefficients or don't drop out coeffs
        
        # use zigzag to get the best coeffs
        coeffs_zigzag = zigzag(win1)
        coeffs_zigzag_clean = coeffs_zigzag
        # print(win1)
        coeffs_zigzag = coeffs_zigzag.reshape((8,8))
        
        # keep the k best coeffs 
        coeffs_best = uppertriangularmask(coeffs_zigzag)
        block_k = inverse_zigzag(coeffs_best, vmax=zBLOCK.shape[0], hmax=zBLOCK.shape[1])
        win_k = idct2(block_k)

        # reconstruct with k-best coefficients
        img_reconstructed_kcoefs[i1:i1+8, i2:i2 +8] = win_k

        # reconstrcut with all coefficients
        block_clean = inverse_zigzag(coeffs_zigzag_clean, vmax=zBLOCK.shape[0], hmax=zBLOCK.shape[1])
        win_clean = idct2(block_clean)
        img_reconstructed_allcoefs[i1:i1 + 8, i2:i2 + 8] = idct2(block_clean)
        

# plot
# fig = plt.figure(figsize=(10,8))
# ax1 = fig.add_subplot(1,2,1)
# ax1.imshow(img_reconstructed_kcoefs,cmap="gray")
# ax1.set_title("Image Reconstructed (K=27) AC")
# ax2 = fig.add_subplot(1,2,2)
# ax2.imshow(img_reconstructed_allcoefs,cmap="gray")
# ax2.set_title("Image Reconstructed All Coeffs")
# plt.show()

# ==================================================== #
# 2.3 Watermark
#           NOTE: I combined task 3 and 4
#                 by directly embedding the watermark
#                 in the upper triangular mask matrix
#                 a random seed was used for reproducability

# get the uppertriangular matrix of 1s and 0s
ut_mask = getuppertriangularmask()

def applywatermark(ut_block, coeffs_wm,alpha=0.1):

    # seed for consistent results
    np.random.seed(42)
    
    for a,b in np.nditer([ut_block,coeffs_wm], op_flags=['readwrite']):
        # check if there is a 1 the uppertraingular matrix 
        if(a==1): 
            # add the watermark using the following formula: 
            # c_i' = c_i * (1 + alpha * w_i)
            wm = np.random.normal(loc=0, scale=1)
            b[...] = b * (1 + alpha * wm)
    
    return coeffs_wm.flatten()

# ==================================================== #
#   2.5 Replace the k-non dct coefficients 

img2    = iio.imread("images/lena512.bmp").astype(float)
height  = len(img2[1])            # one column of image
width   = len(img2[0])            # one row of image
im      = img2


# dct 
dct_domain  = np.zeros_like(img2)
dct_wm      = np.zeros((64,len(np.arange(1,width,8)) * len(np.arange(1,height,8))))
dct_reconstrcuted = np.zeros_like(img2)

# dct 
dct_diff = np.zeros_like(img2)

# reconstructed
img_reconstructed_wm        = np.zeros_like(img2)
img_original                = img_reconstructed_allcoefs
img_reconstructed_diff      = np.zeros_like(img2)

# sliding window..
for i1 in np.arange(0,width,8).reshape(-1):
    for i2 in np.arange(0,height,8).reshape(-1):
        zBLOCK = im[i1:i1+8, i2:i2 +8]
      
        # Forward Discrete Cosine Transform
        win1 = dct2(zBLOCK)
        dct_domain[i1:i1+8, i2:i2 +8] = win1

        #  Embedd the watermark by replacing c_i with c_i'
        
        # use zigzag to order DCT coefficiennts 
        coeffs_zigzag = zigzag(win1)
        coeffs_zigzag = coeffs_zigzag.reshape((8,8))
        
        # embedd watermark 
        ut_block = getuppertriangularmask()
        ut_block = ut_block.astype('float')
        coeffs_wm = applywatermark(ut_block,coeffs_zigzag)

        # watermarked block in dct
        block_wm = inverse_zigzag(coeffs_wm, vmax=zBLOCK.shape[0], hmax=zBLOCK.shape[1])
        
        # differnce in dct (original - watermarked)
        block_diff = win1 - block_wm 
        dct_diff[i1:i1+8, i2:i2 +8] = block_diff

        # reconstruct watermarked image
        win_wm = idct2(block_wm)
        img_reconstructed_wm[i1:i1+8, i2:i2 +8] = win_wm 

        # reconstruct difference image
        win_diff = idct2(block_diff)
        img_reconstructed_diff[i1:i1+8, i2:i2 +8] = win_diff

# spatial difference iamge
img_spatial_diff = np.absolute(img_original - img_reconstructed_wm)

# plot
fig = plt.figure(figsize=(10,8))
# # ax1 = fig.add_subplot(2,2,1)
# # ax1.imshow(img_original,cmap='gray')
# # ax1.set_title('Original Image')
# # ax2 = fig.add_subplot(2,2,2)
# # ax2.imshow(img_reconstructed_wm, cmap='gray')
# # ax2.set_title("Watermarked Image (alpha=0.1)")
# ax1 = fig.add_subplot(1,2,1)
# ax1.imshow(img_reconstructed_diff, cmap='gray')
# ax1.set_title("Difference Image K=2 (DCT)")
# ax2 = fig.add_subplot(1,2,2)
# ax2.imshow(img_spatial_diff, cmap='gray')
# ax2.set_title("Difference Image K=2 (Spatial)")
# plt.show()

# fig = plt.figure(figsize=(10,8))
# ax1 = fig.add_subplot(1,1,1)
# ax1.imshow(img_original,cmap='gray')
# ax1.set_title('Original Image')
# plt.show()

# ==================================================== #
#   2.7 Display the histograms of the difference image, original and watermarked image

plt.subplot(1,3,1)
plt.hist(img_reconstructed_allcoefs.flatten(), bins=256, range=[0, 256], alpha=0.5)
plt.title('Original')
plt.subplot(1,3,2)
plt.hist(img_reconstructed_wm.flatten(), bins=256, range=[0, 256], alpha=0.5)
plt.title('Watermarked')
plt.subplot(1,3,3)
plt.hist(img_reconstructed_diff.flatten(), bins=256, range=[0, 256], alpha=0.5)
plt.title('Difference')
plt.show()

