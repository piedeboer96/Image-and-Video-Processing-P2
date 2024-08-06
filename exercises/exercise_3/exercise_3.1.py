import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

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


# NOTE: PREPROCESSING
#   1. get a single flat vector from each k^2 dimenisonal matrix
#      which gives us a list of K^2 length of grayscale intensities
#   2. concatenate all K^2 for each subject into Z rows
#      to form a matrix M that represent the full dataset
#  NOTE: WHAT IS PCA:
#      Statististical procedure that allows you to summarize 
#      the information content in large data tables by means
#      of a smaller set of "summary indices"
#      that can be more easily visualized an summarize

# STEP 0:
#   read all images in grayscale and resize each image
#   to build the main matrix M that contains entire dataset

M = []
path = 'dataset'


subjects = 10
subjectimages = 10
k = 100 # based on dimension image
z = 100
result_arr = np.zeros((100,10000))
N = 100

count_z =0

for i in range(subjects):
    for j in range(subjects):
        
        # get path and resize
        temp = path + '/s' + str(i+1) + '/' + str(j) + '.jpg'

        # TODO: check if this resize method is sufficiently...
        img = cv2.imread(temp,0)
        img = cv2.resize(img, (100,100), interpolation=cv2.INTER_CUBIC)
    
        # each image flattened
        arr_k2 = img.flatten()

        result_arr[count_z, :10000] = arr_k2

        count_z = count_z + 1

# make it a numpy array 
# result_arr = np.stack(result_arr, axis=0)
M = result_arr

print('shape of M', M.shape)

# NOTE:
#   the entire data set is now contained in one array
#   thus, let's apply PCA 
#   based on procedure provided in the link

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

#   step 6: 
#       take the top N eigenvectors 
#       with largest corresponding eigenvalue magnitude
eigenVectors_top_N = eigenVectors[:,idx[:N]]

# NOTE: extra stuff
# # Make a list of (eigenvalue, eigenvector) tuples
# eig_vals = eigenValues; eig_vecs = eigenVectors
# eig_pais = (eig_vals, eig_vecs)
# eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# # Sort the (eigenvalue, eigenvector) tuples from high to low
# eig_pairs.sort(key=lambda x: x[0], reverse=True)

# # Visually confirm that the list is correctly sorted by decreasing eigenvalues
# print('Eigenvalues in descending order:')
# for i in eig_pairs:
#     print(i[0])

# tot = sum(eig_vals)
# var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
# cum_var_exp = np.cumsum(var_exp)

# with plt.style.context('seaborn-whitegrid'):
#     # plt.figure(figsize=(10,10))

#     plt.bar(range(100), var_exp, alpha=0.5, align='center',
#             label='individual explained variance')
#     plt.step(range(100), cum_var_exp, where='mid',
#              label='cumulative explained variance')
#     plt.ylabel('Explained variance ratio')
#     plt.xlabel('Principal components')
#     plt.legend(loc='best')
#     plt.tight_layout()

# plt.show()


#   step 7:
#       transform input data by projecting 
#       it onto the space created by the top N eigenvectors
#       by taking dot product (projection matrix)

print('top N', eigenVectors_top_N.shape)
# less dimensional
# eigenVectors_top_N = eigenVectors_top_N[:50,:]
print('M', M.shape)

transform = np.dot(eigenVectors_top_N, M)

# # weights for PCA
weights = np.dot(eigenVectors_top_N, M)


# # Reconstruct a face image using the PCA weights and eigenfaces
# reconstructed_face_index = 0  # Index of the face image to reconstruct
# reconstructed_face_weights = weights[reconstructed_face_index]
# reconstructed_face = np.dot(reconstructed_face_weights, eigenVectors_top_N.T) + mean_cols

# # Reshape the reconstructed face to its original dimensions
# reconstructed_face = reconstructed_face.reshape((k, k))

# # Display the reconstructed face
# plt.imshow(reconstructed_face, cmap='gray')
# plt.axis('off')
# plt.show()




# reshape to get the eigenfaces 
eigenfaces = transform.reshape(N, 100, 100)

# NOTE:
#   'tresholding' and variance... 
#    determines principle components

# plot the eigenfaces
fig, axs = plt.subplots(nrows=10,ncols=10, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    eigenface = eigenfaces[i]
    # print(eigenface)
    ax.imshow(eigenface,cmap='gray') 
    ax.axis('off')
plt.show()

# ============================================ #
#   3.2 Reconstruction Task 
#       TODO: Given our eigenface vectors, we can represent a new face by taking the dot product between the (flattened) input face image and the N eigenfaces.
#       This allows us to represent each face as a linear combination of principal components:

print('eigenface', eigenfaces[0].shape)
print('weight', weights[0].shape)

print(weights[0])

# ============================================ #
#   3.3 Variance between reconstructions
#       
#       using more eigenfaces, we capture more variance
#       thus more details, though possible overfit
#       
#       NOTE:
#           best way is probably by working with the eigenvalues
#           so, we need the correspoding the eigenvalues




# ============================================ #
#   3.4 Reconstructing with 'wrong' weights
#       
#       using more eigenfaces, we capture more variance
#       thus more details, though possible overfit




