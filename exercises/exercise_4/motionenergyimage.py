import numpy as np

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class MotionEnergyImage:
    @staticmethod
    def mei(w,path_to_video,id):
    
        # =======================================================
        #   step 1: get MEIs

        # optical flow
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        color = np.random.randint(0, 255, (100, 3))
        cap = cv.VideoCapture(path_to_video)

        # Take the first frame and find corners in it
        ret, old_frame = cap.read()
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        frames_buffer = []
        mask = np.zeros_like(old_frame)
        cumulative_motion_image = np.zeros_like(old_gray)

        # list for the motion energy images
        MEI = []

        # frame count
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print('No frames grabbed!')
                break

            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Calculate optical flow
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

            img = cv.add(frame, mask)

            # Calculate the absolute difference between the current frame and the previous frame
            diff = cv.absdiff(frame_gray, old_gray)

            # Update the cumulative motion image every w frames
            if frame_count % w == 0:
                cumulative_motion_image = cv.bitwise_or(cumulative_motion_image, diff)
                
                # binarize in or der to build the MEI
                binarized_image = cv.threshold(cumulative_motion_image, 50, 255, cv.THRESH_BINARY)[1]
                MEI.append(binarized_image)
                cv.imwrite(f'cumulative_motion_image_{frame_count // w}.jpg', binarized_image)

            cv.imshow('frame', img)
            cv.imshow('cumulative_motion_image', cumulative_motion_image)

            k = cv.waitKey(30) & 0xff
            if k == 27:
                break

            # Update the previous frame and points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            frame_count += 1

        cv.destroyAllWindows()

        # raw MEIs
        MEI_raw = MEI.copy()

        # =======================================================
        #   step 2: apply morphologic operations to MEIs
        kernel = np.ones((3,3),np.uint8)

        MEI_cleaned = MEI.copy()

        # NOTE: just show 3 types.. and therfore which is coolest!
        for i in range(15):
            #temp = cv.morphologyEx(MEI[i], cv.MORPH_OPEN, kernel)
            temp = cv.dilate(MEI[i],kernel,iterations = 1)
            MEI_cleaned[i] = temp

        # =======================================================
        #   step 3: find outline to images

        print("CLEANED MEI", MEI_cleaned[i].shape)

        MEI_contours = []
        for i in range(15):
            img = MEI_cleaned[i]
            # print("CLEANED MEI", img.shape)
            ret, thresh = cv.threshold(img, 80, 255, 0)
            contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            MEI_contours.append(contours)

        MEI_contour_imgs = []

        for i in range(15):

            contour_img = cv.drawContours(np.zeros_like(MEI_cleaned[i]), MEI_contours[i], -1, (255, 255, 255), 1)
            MEI_contour_imgs.append(contour_img)

        # # ============================================ #
        # #   step 4: Shape descriptors/
        # #           Hu moments for the MEI outlines
        # #           including intra video
        # #           and inter-video frame comparrisons
        # #           which defines actions.. 
        # #           through means of MSE
        # #           Hu moments is done by means of 
        # #           built-in openCV methods


        # # list of arrays of 'hu moments' for first dancer/
        # # all the shape descriptors for the first dancer
        # # which represent 'action 1' / breakdance move 1 
        action_sds = []

        # print(type(MEI_contours[0]))

        for i in range(15):
            temp = MEI_contour_imgs[i]
            temp_np = np.array(temp)  
            moments = cv.moments(temp_np)
            hu_moments = cv.HuMoments(moments).flatten()
            action_sds.append(hu_moments)
            # print(hu_moments)

        print('Succesfully Executed MEI')
        
        return MEI_raw, MEI_cleaned, MEI_contours, MEI_contour_imgs, action_sds

    