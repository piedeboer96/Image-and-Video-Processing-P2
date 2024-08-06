import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import imageio as iio
import scipy


from statistics import mean
from numpy import pi
from numpy import r_
from scipy import signal


"""
    @TITLE:
        DCT Domain Watermark Detection (Decoding)
    @RECOURCES:
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
def getuppertriangularmask():
    
    # k=15-1 
    # mask = np.array([[0, 1, 1, 1, 1, 0, 0, 0],
    #                  [1, 1, 1, 1, 0, 0, 0, 0],
    #                  [1, 1, 1, 0, 0, 0, 0, 0],
    #                  [1, 1, 0, 0, 0, 0, 0, 0],
    #                  [1, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0]]) 
    
    mask = np.array([[0, 1, 1, 1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 1, 0, 0, 0],
                     [1, 1, 1, 1, 0, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0]])  

    return mask  
def uppertriangularmask(input):

    # k=15-1 
    # mask = np.array([[0, 1, 1, 1, 1, 0, 0, 0],
    #                  [1, 1, 1, 1, 0, 0, 0, 0],
    #                  [1, 1, 1, 0, 0, 0, 0, 0],
    #                  [1, 1, 0, 0, 0, 0, 0, 0],
    #                  [1, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0]])

    # k=28-1
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
    # mask = np.array([[0, 1, 1, 1, 1, 0, 0, 0],
    #                  [1, 1, 1, 1, 0, 0, 0, 0],
    #                  [1, 1, 1, 0, 0, 0, 0, 0],
    #                  [1, 1, 0, 0, 0, 0, 0, 0],
    #                  [1, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 0]]) 
    
    mask = np.array([[0, 1, 1, 1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 1, 0, 0, 0],
                     [1, 1, 1, 1, 0, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0]])  

    return mask  
def applywatermark(ut_block, coeffs_wm,mean,variance,alpha):

    # seed for consisten results
    np.random.seed(42)
    
    for a,b in np.nditer([ut_block,coeffs_wm], op_flags=['readwrite']):
        # check if there is a 1 the uppertraingular matrix 
        if(a==1): 
            # add the watermark using the following formula: 
            # c_i' = c_i * (1 + alpha * w_i)
            wm = np.random.normal(loc=mean, scale=variance)
            b[...] = b * (1 + alpha * wm)
            a[...] = wm
    
    watermark = ut_block

    return coeffs_wm.flatten(), watermark
def createwatermarkedimage(img,mean,variance,alpha):

    img2    = img; 
    height  = len(img2[1])            
    width   = len(img2[0])          
    im      = img2

    # dct 
    dct_domain  = np.zeros_like(img2)

    # full watermark
    watermark = np.zeros_like(img2)

    # reconstructed watermarked image
    img_reconstructed_wm  = np.zeros_like(img2)

    # sliding window block size
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
            coeffs_wm, wm = applywatermark(ut_block,coeffs_zigzag,mean,variance,alpha)
            
            # full watermark matrix 
            watermark[i1:i1+8, i2:i2+8] = wm

            # watermarked block in dct
            block_wm = inverse_zigzag(coeffs_wm, vmax=zBLOCK.shape[0], hmax=zBLOCK.shape[1])

            # reconstruct watermarked image
            win_wm = idct2(block_wm)
            img_reconstructed_wm[i1:i1+8, i2:i2 +8] = win_wm 

    return img_reconstructed_wm, watermark
def keep_k_coeffs(img):
    img2    = img; 
    height  = len(img2[1])            
    width   = len(img2[0])          
    im      = img2

    # dct 
    dct_domain  = np.zeros_like(img2)

    # reconstructed largest non-DC DCT coefficients image
    img_reconstructed_kcoefs = np.zeros_like(img2)

    # sliding window block size
    for i1 in np.arange(0,width,8).reshape(-1):
        for i2 in np.arange(0,height,8).reshape(-1):
            zBLOCK = im[i1:i1+8, i2:i2 +8]
      
            # Forward Discrete Cosine Transform
            win1 = dct2(zBLOCK)
            dct_domain[i1:i1+8, i2:i2 +8] = win1

            # use zigzag to get the best coeffs
            coeffs_zigzag = zigzag(win1)

            # print(win1)
            coeffs_zigzag = coeffs_zigzag.reshape((8,8))
        
            # keep the k best coeffs 
            coeffs_best = uppertriangularmask(coeffs_zigzag)
            block_k = inverse_zigzag(coeffs_best, vmax=zBLOCK.shape[0], hmax=zBLOCK.shape[1])
            win_k = idct2(block_k)

            # reconstruct with k-best coefficients
            img_reconstructed_kcoefs[i1:i1+8, i2:i2 +8] = win_k

    return img_reconstructed_kcoefs
def estimationwatermark(original, watermarked, alpha):
    
    img2    = original
    height  = len(img2[1])            
    width   = len(img2[0])   

    # uppertriangular mask corresponding to k-value
    ut_mask = getuppertriangularmask()
    ut_mask = ut_mask.astype('float')

    # rename
    img_og  = original
    img_wm  = watermarked

    # estimation matrix
    w_hat = np.zeros_like(img2)

    # sliding windows 
    for i1 in np.arange(0,width,8).reshape(-1):
        for i2 in np.arange(0,height,8).reshape(-1):
           
            # spatial block
            zBLOCK_og = img_og[i1:i1+8, i2:i2 +8]
            zBLOCK_wm = img_wm[i1:i1+8, i2:i2 +8]
      
            # Forward Discrete Cosine Transform
            win_og = dct2(zBLOCK_og)
            win_wm = dct2(zBLOCK_wm)
            
            # zigzag, cause watermark was applied to most important ones
            win_og = zigzag(win_og).reshape((8,8))
            win_wm = zigzag(win_wm).reshape((8,8))

            # only for the k entries estimation
            win_og = ut_mask * win_og
            win_wm = ut_mask * win_wm 

            # copmute estimation for the specific block
            coeff_diff  = win_og - win_wm
            coeff_alpha = alpha * win_og

            #temp = np.divide(coeff_diff, coeff_alpha)

            temp = np.divide(coeff_diff, coeff_alpha, out=np.zeros_like(coeff_diff), where=coeff_alpha != 0)

            # compute estimation for block and construct for entire
            w_hat[i1:i1+8, i2:i2 +8] = temp.astype('float')

    # replace nan_values 
    w_hat[np.isnan(w_hat)] = 0
    return w_hat

"""
    Main
"""

# ==================================================== #
#   0. Create two mystery images

img = iio.imread("images/lena512.bmp").astype(float)
img_other = iio.imread("images/barbara512.bmp").astype(float)
img_wm_1, wm_1 = createwatermarkedimage(img,mean=0,variance=1,alpha=0.5)
img_wm_2, wm_2 = createwatermarkedimage(img_other,mean=0,variance=0.05,alpha=0.5)


# fig = plt.figure(figsize=(10,8))
# ax1 = fig.add_subplot(2,2,1)
# ax1.imshow(img_wm_1,cmap='gray')
# ax1.set_title('Mystery Image 1')
# ax2 = fig.add_subplot(2,2,2)
# ax2.imshow(img_wm_2,cmap='gray')
# ax2.set_title('Mystery Image 2')
# plt.show()

# ==================================================== #
#   1. Compute the 2D blockwise DCT of the mystery image ('full image'
#       # TODO: add this in exercise 2.1

# array for dct mystery image 1 and 2
imsize = img_wm_1.shape
dct_img_1 = np.zeros(imsize)
dct_img_2 = np.zeros(imsize)

for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        dct_img_1[i:i+8,j:j+8] = dct2( img_wm_1[i:i+8,j:j+8] )
        dct_img_2[i:i+8,j:j+8] = dct2( img_wm_2[i:i+8,j:j+8] )

dct_img = dct2(img)

# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(dct_img_1,vmax = np.max(dct_img_1)*0.01,vmin = 0)
# plt.title( "8x8 DCTs of the mystery image 1")
# plt.subplot(1,2,2)
# plt.imshow(dct_img_2,vmax = np.max(dct_img_2)*0.01,vmin = 0)
# plt.title( "8x8 DCTs of the mystery image 2")
# plt.show()

# ==================================================== #
#   2. Use zig-zag and use a upper triangular matrix 
#      to keep most important coeffs
#      ofcourse again based on K before (no)

img_1_kcoeffs = keep_k_coeffs(img_wm_1)
img_2_kcoeffs = keep_k_coeffs(img_wm_2)

# ==================================================== #
#   3. Estimation of the watermark 
#      to keep most important coeffs
#      ofcourse again based on K before (no)

#      we use the original c_i's
#      according to the book we want to find out if that
#      is the watermarked image in question (page 629)

# compute estimations for both mystery images
wm_1_approx = estimationwatermark(original=img, watermarked=img_wm_1, alpha=0.5)
wm_2_approx = estimationwatermark(original=img, watermarked=img_wm_2, alpha=0.5)

# ==================================================== #
#   4. Measure similarity using correlation coefficient       
#           - gamma computed for each sldiding window
#           - first we multiply with upper triangluar, represent k
#             keep the non-zeros, flatten
#             do the computation of gamma 
#             for each sliding block

#wm_1_approx = -1*wm_2_approx

# img dim
width = 512
height = 512

# method to compute gamma, based for sliding window
def gamma(wm_k, wma_k, wm_mean,wma_mean):
    
    numerator = 0
    
    denom_1 = 0 
    denom_2 = 0 


    if len(wm_k)!=27 or len(wma_k) != 27:
        print('worng K ')
        print('len wm_k', len(wm_k))
        print('len wma_k', len(wma_k))
        return 0 

    for idx, x in enumerate(wm_k):
        numerator += (wma_k[idx] - wma_mean)*(wm_k[idx] - wm_mean)
        
        denom_1 += (wma_k[idx] - wma_mean)**2
        denom_2 += (wm_k[idx] - wm_mean)**2
    
    denom = np.sqrt((denom_1 * denom_2))

    if denom==0:
        return 0

    return numerator/denom

# watermark 1 -- original & approximated (means)
wm_1_mean = np.mean(wm_1)
wma_1_mean = np.mean(wm_1_approx)

# watermark 2 -- original & approximated (means)
wm_2_mean = np.mean(wm_2)
wma_2_mean = np.mean(wm_2_approx)

# gammvalues
gammavalues_1 = []
gammavalues_2 = []

for i1 in np.arange(0,width,8).reshape(-1):
    for i2 in np.arange(0,height,8).reshape(-1):

        block_wm_1  = wm_1[i1:i1+8, i2:i2 +8]
        block_wma_1 = wm_1_approx[i1:i1+8, i2:i2 +8]

        block_wm_2  = wm_2[i1:i1+8, i2:i2 +8]
        block_wma_2 = wm_2_approx[i1:i1+8, i2:i2 +8]

        # non-zero correspond to k
        wm_1_k =  block_wm_1[block_wm_1!=0].flatten()
        wma_1_k = block_wma_1[block_wma_1!=0].flatten()
        
        wm_2_k =  block_wm_2[block_wm_2!=0].flatten()
        wma_2_k = block_wma_2[block_wma_2!=0].flatten()

        # gamma computation
        gammavalue_1 = gamma(wm_1_k,wma_1_k,wm_1_mean,wma_1_mean)
        gammavalues_1.append(gammavalue_1)

        gammavalue_2 = gamma(wm_2_k,wma_2_k,wm_2_mean,wma_2_mean)
        gammavalues_2.append(gammavalue_2)
        


# decide gamma based on mean
gamma_1 = np.abs(mean(gammavalues_1))

# get max and mix gamma value...
gamma_2 = np.abs(mean(gammavalues_2))

print('Watermark 1 Anlaysis')
print('gamma 1', gamma_1)

print('Watermark 2 Anlaysis')
print('gamma 2', gamma_2)

# plot histograms for investigation
plt.subplot(1,2,1)
plt.hist(wm_1.flatten(), bins=256, range=[0, 20], alpha=0.5)
plt.title('Watermark 1 - Original')
plt.subplot(1,2,2)
plt.hist(wm_1_approx.flatten(), bins=256, range=[0, 20], alpha=0.5)
plt.title('Watermark 1 - Approximated')
plt.show()

plt.subplot(1,2,1)
plt.hist(wm_2.flatten(), bins=256, range=[0, 20], alpha=0.5)
plt.title('Watermark 2 - Original')
plt.subplot(1,2,2)
plt.hist(wm_2_approx.flatten(), bins=256, range=[0, 20], alpha=0.5)
plt.title('Watermark 2 - Approximated')
plt.show()

# ==================================================== #
#   5. Decision procedure about watermarked
#      based on similarity measure gamma

#   NOTE:  
#       we computed gamma above...
#       for fun i took two complete different source images
#       to be watermarked...
#       the correlation min,mean,max became very low... =)


T = 0.98
