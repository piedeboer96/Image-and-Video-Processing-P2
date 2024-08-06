import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from pcpie import PcPie

from sklearn.decomposition import PCA
from numpy.linalg import eig


""" 
    @TITLE:
        Eigenfaces using Principal Component Analysis

    @RECOURCES:
        https://machinelearningmastery.com/face-recognition-using-principal-component-analysis/
        https://pyimagesearch.com/2021/05/10/opencv-eigenfaces-for-face-recognition/
        https://www.sartorius.com/en/knowledge/science-snippets/what-is-principal-component-analysis-pca-and-how-it-is-used-507186
        https://bic-berkeley.github.io/psych-214-fall-2016/subtract_means.html
        https://pythonnumericalmethods.berkeley.edu/notebooks/chapter15.04-Eigenvalues-and-Eigenvectors-in-Python.html
        https://stackoverflow.com/questions/8092920/sort-eigenvalues-and-associated-eigenvectors-after-using-numpy-linalg-eig-in-pyt
        https://towardsdatascience.com/pca-eigenvectors-and-eigenvalues-1f968bc6777a
        https://wiki.pathmind.com/eigenvector
        https://learnopencv.com/face-reconstruction-using-eigenfaces-cpp-python/
        http://www.diva-portal.org/smash/get/diva2:1413368/FULLTEXT01.pdf
        https://www.cs.toronto.edu/~guerzhoy/320/lec/pca.pdf
        https://en.wikipedia.org/wiki/Eigenface
        https://en.wikipedia.org/wiki/Principal_component_analysis
        https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html
"""

# DATASET
# ============================================ #
#       - read all images in grayscale 
#       - resize each image
#       - build matrix M that contains entire dataset
#       - display contents of matrix M

# information M 
M = []
# path = 'dataset'
# path = 'dataset2'
path = 'dataset3'
subjects = 10
subjectimages = 10
rows = subjects
columns = subjectimages

# resize dimension
k = 100 
z = 100
N = 100
result_arr = np.zeros((100,10000))

# resize
count_z =0
for i in range(subjects):
    for j in range(subjects):
        
        # path
        temp = path + '/s' + str(i+1) + '/' + str(j) + '.jpg'

        # resize
        img = cv2.imread(temp,0)
        img = cv2.resize(img, (100,100), interpolation=cv2.INTER_CUBIC)
    
        # flatten image
        arr_k2 = img.flatten()
        result_arr[count_z, :10000] = arr_k2
        count_z = count_z + 1

M = result_arr

# visualize dataset
fig, axs = plt.subplots(rows, columns, figsize=(10, 10))
for i in range(subjects):
    for j in range(subjectimages):
        # reshape 
        img = M[i * subjects + j].reshape(k, z)
        
        # display 
        axs[i, j].imshow(img, cmap='gray')
        axs[i, j].axis('off')
plt.tight_layout()
plt.show()

print(M.shape)


# PRINCIPLE COMPONENT ANALYSIS
# ============================================ #
#   step 1:
#       copmute mean of each column in the matrix
#       giving average pixel intensity for every (x,y)-coordinate
#       in the image dataset
mean_cols = np.mean(M, axis=0)

#   step 2:
#       subract the mean from each column c_i
M_meancentered = M - mean_cols

#   step 3:
#       compute the covariance matrix
M_cov = np.cov(M_meancentered)

#   step 4:
#       perform an eigenvalue decomposition on 
#       the covariance matrix to get eigenvalues 
#       and eigenvetctors
eigenValues,eigenVectors = np.linalg.eig(M_cov)

#   step 5:
#       sort eigenvectors (v) by absolute value of
idx = np.argsort(np.abs(eigenValues))[::-1]
eigenVectors = eigenVectors[:,idx]

#  step 6: 
#       take the top N eigenvectors 
#       with largest corresponding eigenvalue magnitude
eigenVectors_top_N = eigenVectors[:,idx[:N]]

#   step 7:
#       transform input data by projecting 
#       it onto the space created by the top N eigenvectors
#       by taking dot product (projection matrix)
transform = np.dot(eigenVectors_top_N, M)

# reshape to get the eigenfaces 
# NOTE:
#   this messes up the the position, 
#   compared to the original M plot and the new eigenfaces... 
eigenfaces = transform.reshape(N, 100, 100)


# egienfaces (= top eigenvectors projected)
eigenfaces = np.dot(eigenVectors_top_N.T, M_meancentered).reshape(N, k, z)

print('eigenVectors...', (eigenVectors_top_N.T).shape)

# eigenfaces sorted
fig, axs = plt.subplots(nrows=rows, ncols=columns, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    eigenface = eigenfaces[i]
    ax.imshow(eigenface, cmap='gray')
    ax.axis('off')
plt.title('Eigenfaces')
plt.tight_layout()
plt.show()


# first 10 principle components
fig, axs = plt.subplots(2, 5, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    eigenface = eigenfaces[i]
    var = np.round(np.var(eigenface), 1)
    ax.imshow(eigenface, cmap='jet')
    title = str(i) + 'th ' r'$\sigma^2$: ' + str(var)
    ax.set_title(title)
    ax.axis('off')
fig.suptitle('First 10 Principal Components (Eigenfaces)', y=1.05)
plt.tight_layout()
plt.show()

# mean face
mean_face = np.mean(M, axis=0)
mean_face = np.reshape(mean_face, (100,100))


# # fig = plt.figure(figsize=(10,8))
# ax1 = fig.add_subplot(1,2,1)
# ax1.imshow(mean_face, cmap='gray')
# ax1.set_title("Mean Face (Gray)")
# title = "Mean Face " r'$\sigma^2$: ' + str(np.round(np.var(mean_face),1 ))
# ax1.set_title(title)
# ax2 = fig.add_subplot(1,2,2)
# ax2.imshow(mean_face, cmap='jet')
# ax2.set_title(title)
# plt.show()


# RECONSTRUCTION 
# ================================================= #
# Reconstruction Task 
#       method 1 'acquires': 
#           - mean_face
#           - eigenfaces
#           - weights
#       method 2 'assemble':
#           - builds face using (correct) weights

# person
subject_id = 1

# =================================================
# 2 eigenfaces
num_components = 2

idx_start = (subject_id - 1) * 10
idx_end = idx_start + num_components
subject_images = np.array(M)[idx_start:idx_end]
w2, ef2, fm2 = PcPie.pcacool(subject_images,num_components)
face_2 = PcPie.assemble(fm2,ef2,w2,num_components,False)
face_2_wrong = PcPie.assemble(fm2,ef2,w2,num_components,True)

# =================================================
# all eigenfaces

num_components = 10
idx_start = (subject_id - 1) * 10
idx_end = idx_start + num_components
si_10 = np.array(M)[idx_start:idx_end]
w10, ef10, fm10 = PcPie.pcacool(si_10,num_components)
face_10 = PcPie.assemble(fm10, ef10, w10,num_components,False)
face_10_wrong = PcPie.assemble(fm10, ef10, w10,num_components,wrong=True)

# ================================================
# VARIANCE
var_face_2 = np.round(np.var(face_2))
var_face_2_wrong = np.round(np.var(face_2_wrong))
var_face_10 = np.round(np.var(face_10))
var_face_10_wrong = np.round(np.var(face_10_wrong))


# =================================================
# PLOTS 
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(np.reshape(fm2,(100,100)), cmap='gray')
ax1.set_title('Mean Face (2 Faces)')
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(np.reshape(fm10,(100,100)), cmap='gray')
ax2.set_title('Mean Face (All Faces)')
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(face_2, cmap='gray')
ax1.set_title('Original $\\alpha_i$ (2 Faces) $\\sigma^2=$ {:.2e}'.format(var_face_2))
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(face_2_wrong, cmap='gray')
ax2.set_title('Modified $\\alpha_i$ (2 Faces) $\\sigma^2=$ {:.2e}'.format(var_face_2_wrong))
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(face_10, cmap='gray')
ax1.set_title('Original $\\alpha_i$ (All Faces) $\\sigma^2=$ {:.2e}'.format(var_face_10))
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(face_10_wrong, cmap='gray')
ax2.set_title('Modified $\\alpha_i$ (All Faces) $\\sigma^2=$ {:.2e}'.format(var_face_10_wrong))
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(face_10, cmap='jet')
ax3.set_title('Original $\\alpha_i$ (All Faces) $\\sigma^2=$ {:.2e}'.format(var_face_10))
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(face_10_wrong, cmap='jet')
ax4.set_title('Modified $\\alpha_i$ (All Faces) $\\sigma^2=$ {:.2e}'.format(var_face_10_wrong))
plt.tight_layout()
plt.show()

# fig = plt.figure(figsize=(10,8))
# ax1 = fig.add_subplot(1,2,1)
# ax1.imshow(face_2, cmap='jet')
# ax1.set_title('Oringal Weights (2)')
# ax2 = fig.add_subplot(1,2,2)
# ax2.imshow(face_10_wrong, cmap='jet')
# ax2.set_title('Modified Weights (2)')
# plt.show()


# fig = plt.figure(figsize=(10,8))
# ax1 = fig.add_subplot(1,2,1)
# ax1.imshow(face_2, cmap='jet')
# title1 = "2 Faces, " + str(np.round(np.var(face_2),0))
# ax1.set_title(title1)
# ax2 = fig.add_subplot(1,2,2)
# ax2.imshow(face_10, cmap='jet')
# title2 = "10 Faces, " + str(np.round(np.var(face_10),0))
# ax2.set_title(title1)
# plt.show()

# ============================================= #
# 95% VARIANCE THEORY
# NOTE: THIS CODE IS COPIED FROM INTERNET
#       AND ADAPTED TO WORK FOR THIS PROJECT!
      
# sum of eigenvalues
eigsum = np.sum(eigenValues)

# calculate the cumulative sum and find the number of components for 95% variance
csum = 0
k95 = 0
for i in range(len(eigenValues)):
    csum += eigenValues[i]
    tv = csum / eigsum
    if tv > 0.95:
        k95 = i + 1
        break

# sum of eigenvalues
eigsum = np.sum(eigenValues)

print("Number of components for 95% variance:", k95)

# explained variance ratio
var_exp = [(i / eigsum) * 100 for i in eigenValues]
cum_var_exp = np.cumsum(var_exp)

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(len(eigenValues)), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(len(eigenValues)), cum_var_exp, where='mid',
             label='cumulative explained variance')

    # vertical dotted red line for 95% variance point
    k95 = np.argmax(cum_var_exp >= 95) + 1
    plt.axvline(x=k95 - 0.5, color='red', linestyle='dotted')

    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title('Dataset 1 (WorldFace)') 
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
