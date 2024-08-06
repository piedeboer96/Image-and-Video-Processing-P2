import numpy as np
import cv2 as cv

"""
    @TITLE:
        Motion Energy Images

    @RECOURSES:
        https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
        https://stackoverflow.com/questions/18954889/how-to-process-images-of-a-video-frame-by-frame-in-video-streaming-using-openc
        https://en.wikipedia.org/wiki/Optical_flow
        https://courses.cs.washington.edu/courses/cse576/22sp/notes/14_Motion_21.pdf
        https://github.com/opencv/opencv/blob/3.1.0/samples/python/opt_flow.py#L24-L34
        http://www.sefidian.com/2019/12/16/a-tutorial-on-motion-estimation-with-optical-flow-with-python-implementation/
"""

# NOTE: adapted starter code form OPENCV
#       which is adapted to capture 'w' amount of frames
#       including drawings of optical flow being exported
#       it uses ShiTomasi corner detection
#       to 'captures' the good points
#       then Lucas-Kanade is used to cacluate optical flow


# TODO:
#   - adapt this method such that every w-frames, we export an image, with the 'drag',
#   drawn optical flow, of the last w-frames
#   and add those images to a plot
#   once the plot has 5 images, STOP 
def calculateopticalflow(path, w):
    cap = cv.VideoCapture(path)
    
    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    # Create a list to store the frames
    frames = []

    # Frame count
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

        # Combine the frame and mask
        img = cv.add(frame, mask)

        # Show the frame
        cv.imshow('frame', img)

        # Save the frame
        if frame_count % w == 0 and frame_count > 0:
            # Save the frame with tracks
            cv.imwrite("tracks_frame{:03d}.jpg".format(frame_count), img)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        
        # Increment the frame count
        frame_count += 1

cv.destroyAllWindows()



path_dancer = "videos\jump_01.mp4"

# calculateopticalflow(path=path_dancer, w=20)

calculateopticalflow(path=path_dancer, w=10)

