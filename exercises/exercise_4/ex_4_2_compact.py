import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from motionenergyimage import MotionEnergyImage as mp

# action 1: jacks
w1 = 10
path_video_1 = r'C:\Users\piede\Desktop\ivp-project-2\videos\jack_01.mp4'

# action 2: jumps
w2 = 4
path_video_2 = r'C:\Users\piede\Desktop\ivp-project-2\videos\jump_01.mp4'

# get all data
MEI_raw_1, MEI_clean_1, MEI_contours_1, MEI_outlines_1, shape_descriptor_1  = mp.mei(w=w1,path_to_video=path_video_1,id=1)
MEI_raw_2, MEI_clean_2, MEI_contours_2, MEI_outlines_2, shape_descriptor_2  = mp.mei(w=w2,path_to_video=path_video_2,id=2)

print('Done')

# PLOTS
# ===========================================
#       1. Raw Motion Energy Images 

# entries
N=12

# jacks... 
# plt.figure(figsize=(12, 8))
# for i in range(N):
#     ax = plt.subplot(4, 3, i + 1)
#     plt.imshow(MEI_raw_1[i], cmap='gray')
#     plt.title(f'Frame {i + 1}')
#     plt.axis('off')
#     ax.set_xticks([])
#     ax.set_yticks([])

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
# plt.suptitle('Raw Motion Energy Images - Jumping Jacks ($\omega$=10)', fontsize=16, fontweight='bold')

# plt.show()

# jumps
# plt.figure(figsize=(12, 8))
# for i in range(N):
#     ax = plt.subplot(4, 3, i + 1)
#     plt.imshow(MEI_raw_2[i], cmap='gray')
#     plt.title(f'Frame {i + 1}')
#     plt.axis('off')
#     ax.set_xticks([])
#     ax.set_yticks([])

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
# plt.suptitle('Raw Motion Energy Images - Jumps ($\omega$=4)', fontsize=16, fontweight='bold')

# plt.show()

# ===========================================
#       2. Cleaned Motion Energy Images 

# jacks... 
# plt.figure(figsize=(12, 8))
# for i in range(N):
#     ax = plt.subplot(4, 3, i + 1)
#     plt.imshow(MEI_clean_1[i], cmap='gray')
#     plt.title(f'Frame {i + 1}')
#     plt.axis('off')
#     ax.set_xticks([])
#     ax.set_yticks([])

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
# plt.suptitle('Dilated Motion Energy Images - Jumping Jacks ($\omega$=10)', fontsize=16, fontweight='bold')

# plt.show()

# # jumps
# plt.figure(figsize=(12, 8))
# for i in range(N):
#     ax = plt.subplot(4, 3, i + 1)
#     plt.imshow(MEI_clean_2[i], cmap='gray')
#     plt.title(f'Frame {i + 1}')
#     plt.axis('off')
#     ax.set_xticks([])
#     ax.set_yticks([])

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
# plt.suptitle('Dilated Motion Energy Images - Jumps ($\omega$=4)', fontsize=16, fontweight='bold')

# plt.show()


# ===========================================
#       3. Outlines

# jacks
# plt.figure(figsize=(12, 8))
# for i in range(N):
#     ax = plt.subplot(4, 3, i + 1)
#     plt.imshow(MEI_outlines_1[i], cmap='gray')
#     plt.title(f'Frame {i + 1}')
#     plt.axis('off')
#     ax.set_xticks([])
#     ax.set_yticks([])

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
# plt.suptitle('Outlines Motion Energy Images - Jumping Jacks ($\omega$=10)', fontsize=16, fontweight='bold')

# plt.show()

# # jumps
# plt.figure(figsize=(12, 8))
# for i in range(N):
#     ax = plt.subplot(4, 3, i + 1)
#     plt.imshow(MEI_outlines_2[i], cmap='gray')
#     plt.title(f'Frame {i + 1}')
#     plt.axis('off')
#     ax.set_xticks([])
#     ax.set_yticks([])

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
# plt.suptitle('Outlines Motion Energy Images - Jumps ($\omega$=4)', fontsize=16, fontweight='bold')

# plt.show()

# ===========================================
#       4. shape descriptors
hu_1 = np.array(shape_descriptor_1)
hu_2 = np.array(shape_descriptor_2)

print('hu2', hu_2.shape)

print(hu_2)

# ===========================================
#      5. shape descriptors
#           - between frames
#           - between actions

action_1_intra = []
action_2_intra = []

hue_inter = []

for i in range(12):

    temp1 = np.mean((hu_1[i+1]-hu_1[i])**2)
    temp2 = np.mean((hu_2[i+1]-hu_2[i])**2)

    inter = np.mean((hu_1[i]-hu_2[i])**2)

    action_1_intra.append(temp1)
    action_2_intra.append(temp2)


    hue_inter.append(inter)




