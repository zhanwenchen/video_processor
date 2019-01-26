'''
Example code for live video processing
Also multithreaded video processing sample using opencv 3.4

Usage:
   python testcv_mt.py {<video device number>|<video file name>}

   Use this code as a template for live video processing

   Also shows how python threading capabilities can be used
   to organize parallel captured frame processing pipeline
   for smoother playback.

Keyboard shortcuts: (video display window must be selected

   ESC - exit
   space - switch between multi and single threaded processing
   a - adjust contrast, brightness, and gamma
   d - running difference of current and previous image
   e - displays canny edges
   f - displays raw frames
   h - display image hue band
   o - apply a 5x5 "fat plus" opening to the thresholded image
   q - histogram equalize value image
   t - do thresholding
   v - write video output frames to file "vid_out.avi"
'''

# Standard imports. They are core Python libraries without installation.
import time
from multiprocessing.pool import ThreadPool
from collections import deque
import argparse

# Non-standard imports. They have been installed by conda or pip.
import cv2
import numpy as np

# Relative (local) imports. They are defined in your project folder(s).
import video
from common import clock, draw_str, StatValue
from utils import cv2_threshold, cv2_morph_open


# Global variables. They should be SCREAMING_SNAKE_CASE and read-only.
WINDOW_NAME = 'VideoProcessor' # The name on top of the video player GUI
TIMEOUT = 100 # For creating the video capture object

# used to execute process_frame when in non threaded mode
class DummyTask:
    def __init__(self, data):
        self.data = data

    def ready(self):
        return True

    def get(self):
        return self.data

class VideoProcessor:
    '''
    Attributes:
        frame_counter (int):
        show_frames (bool):
        diff_frames (bool):
        show_edges (bool):
    show_hue:
    do_threshold:
    adj_img:
    adj_gam:
    m_open:
    hist_eq:
    vid_frames:
    contrast:
    contrast_slider_max:
    brightness:
    brightness_slider_max:
    gamma:
    gamma_slider_max:
    threshold:
    threshold_slider_max:
    '''
    def __init__(self):
        self.frame_counter = 0
        self.showing_frames = True
        self.showing_diff_frames = False
        self.showing_edges = False
        self.showing_hue = False
        self.thresholding = False
        self.adjusting_img = False
        self.adjusting_gamma = False
        self.doing_morph_open = False
        self.doing_hist_eq = False
        self.is_vid_frames = False
        self.contrast = 128
        self.contrast_slider_max = 255
        self.brightness = 128
        self.brightness_slider_max = 255
        self.gamma = 128
        self.gamma_slider_max = 255
        self.threshold = 128
        self.threshold_slider_max = 255


    # this routine is run each time a new video frame is captured
    def process_frame(self, frame, prevFrame, t0):
        '''
        Args:
            frame (NumPy array of dtype np.uint8 of shape (720, 1280, 3)):
            prev_frame (NumPy array of dtype np.uint8 of shape (720, 1280, 3)):
            t0: (float):

        Return:
        '''
        if self.adj_img:
            # shift value to get actual brightness offset
            new_brightness = self.brightness - 128
            # compute the contrast value from the trackbar setting
            if self.contrast > 127:
                new_contrast = 1+(5*(self.contrast - 128)/128)
            else:
                new_contrast = 1/(1+(5*(128 - self.contrast)/128))
            # adjust brightness and contrast
            frame = ((np.float_(frame)-128) * new_contrast) + 128 + new_brightness
            # compute the gamma value from the trackbar setting
            if self.gamma > 127:
                new_gamma = 1+(2*(self.gamma - 128)/128)
            else:
                new_gamma = 1/(1+(2*(128 - self.gamma)/128))
            # apply the gamma function
            frame = 255 * ((frame / 255) ** (1 / new_gamma))
            # then convert the result back to uint8 after clipping at 0 and 255
            np.clip(frame, 0, 255, out=frame) # inplace with 'out' is a bit faster.
            frame = np.uint8(frame)

        if self.hist_eq:
            # convert image to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]  # separate the channels
            val = cv2.equalizeHist(val)
            hsv = cv2.merge((hue, sat, val))
            del val
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if not self.show_frames and self.show_edges:  # edges alone
            edges = cv2.Canny(frame, 100, 200)
            thisFrame = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB, edges)
        elif SHOW_FRAMES and show_edges:  # edges and frames
            edges = cv2.Canny(frame, 100, 200)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB, edges)
            thisFrame = cv2.add(frame, edges)
        else:  # current frame
            thisFrame = frame.copy()

        if self.do_threshold:
            # create threshold mask
            threshMask = self.get_threshold_mask(frame)
            # apply the mask
            thisFrame = threshMask * thisFrame

        if self.showing_hue:
            # convert image to HSV
            hsv = cv2.cvtColor(thisFrame, cv2.COLOR_BGR2HSV)
            hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]  # separate the channels
            # the maximum hue is 170, so scale it into [0,255]
            h32 = np.float32(hue) * 255 / 170
            sch = np.uint8(np.clip(h32, 0, 255))  # clip at 255 and convert back to uint8
            # apply the opencv builtin hue colormap
            thisFrame = cv2.applyColorMap(sch, cv2.COLORMAP_HSV)

        if diff_frames:
            # compute absolute difference between the current and previous frame
            difframe = cv2.absdiff(thisFrame, prevFrame)
            # save current frame as previous
            prevFrame = thisFrame.copy()
            # set the current frame to the difference image
            thisFrame = difframe.copy()
        else:
            # save current frame as previous
            prevFrame = thisFrame.copy()

        return thisFrame, prevFrame, t0


    def get_threshold_mask(self, frame):
        # mB, mG, mR, _ = np.uint8(cv2.mean(frame))
        B, G, R = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        # _, tB = cv2.threshold(B, mB, 1, cv2.THRESH_BINARY)
        # _, tG = cv2.threshold(G, mG, 1, cv2.THRESH_BINARY)
        # _, tR = cv2.threshold(R, mR, 1, cv2.THRESH_BINARY)
        # _, tB = cv2.threshold(B, threshold, 1, cv2.THRESH_BINARY)
        # _, tG = cv2.threshold(G, threshold, 1, cv2.THRESH_BINARY)
        # _, tR = cv2.threshold(R, threshold, 1, cv2.THRESH_BINARY)
        _, B_thresholded = cv2_threshold(B, self.threshold)
        _, G_thresholded = cv2_threshold(G, self.threshold)
        _, R_thresholded = cv2_threshold(R, self.threshold)
        if self.doing_morph_open:
            # create structuring element for morph ops
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # tB = cv2.morphologyEx(tB, cv2.MORPH_OPEN, se, 1)
            # tG = cv2.morphologyEx(tG, cv2.MORPH_OPEN, se, 1)
            # tR = cv2.morphologyEx(tR, cv2.MORPH_OPEN, se, 1)
            tB = cv2_morph_open(tB, se)
            tG = cv2_morph_open(tG, se)
            tR = cv2_morph_open(tR, se)
        threshold_mask = cv2.merge((B_thresholded, G_thresholded, R_thresholded))

        return threshold_mask


# def on_brightness_trackbar(val):
#     global brightness
#     brightness = val
#
#
# def on_contrast_trackbar(val):
#     global contrast
#     contrast = val
#
#
# def on_gamma_trackbar(val):
#     global gamma
#     gamma = val
#
#
# def on_threshold_trackbar(val):
#     global threshold
#     threshold = val


# create a video capture object
def create_capture(source=0):

    # parse source name (defaults to 0 which is the first USB camera attached)
    # source = str(source).strip()
    # chunks = source.split(':')
    # # handle drive letter ('c:', ...)
    # if len(chunks) > 1 and len(chunks[0]) == 1 and chunks[0].isthreshold():
    #     chunks[1] = chunks[0] + ':' + chunks[1]
    #     del chunks[0]
    #
    # source = chunks[0]
    # try:
    #     source = int(source)
    # except ValueError:
    #     pass
    #
    # params = dict(s.split('=') for s in chunks[1:])

    # video capture object defined on source

    iter = 0
    cap = cv2.VideoCapture(source)
    while (cap is None or not cap.isOpened()) & (iter < TIMEOUT):
        time.sleep(0.1)
        iter = iter + 1
        cap = cv2.VideoCapture(source)

    if iter == TIMEOUT:
        print('camera timed out')
        return None
    else:
        print(iter)

    if 'size' in params:
        w, h = map(int, params['size'].split('x'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', source)
        return None

    return cap

# main program
if __name__ == '__main__':

    # print in the program shell window the text at the beginning of the file
    print(__doc__)

    # if there is no argument in the program invocation default to camera 0
    try:
        fn = sys.argv[1] # the first positional argument of argument type.
    except:
        fn = 0

    # grab initial frame, create window
    cv2.waitKey(1) & 0xFF
    cap = video.create_capture(fn)
    ret, frame = cap.read()
    self.frame_counter += 1
    height, width, channels = frame.shape
    prevFrame = frame.copy()
    cv2.namedWindow(WINDOW_NAME)

    # Create video of Frame sequence -- define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    cols = np.int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rows = np.int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_out = cv2.VideoWriter('vid_out.avi', fourcc, 20.0, (cols, rows))

    # Set up multiprocessing
    threadn = cv2.getNumberOfCPUs()
    pool = ThreadPool(processes=threadn)
    pending = deque()

    threaded_mode = True
    onOff = False

    # initialize time variables
    latency = StatValue()
    frame_interval = StatValue()
    last_frame_time = clock()

    # main program loop
    while True:
        while len(pending) > 0 and pending[0].ready():  # there are frames in the queue
            res, prevFrame, t0 = pending.popleft().get()
            latency.update(clock() - t0)
            # plot info on threading and timing on the current image
            # comment out the next 3 lines to skip the plotting
            draw_str(res, (20, 20), "threaded      :  " + str(threaded_mode))
            draw_str(res, (20, 40), "latency        :  %.1f ms" % (latency.value * 1000))
            draw_str(res, (20, 60), "frame interval :  %.1f ms" % (frame_interval.value * 1000))
            # write output video frame
            if vid_frames:
                vid_out.write(res)
            # show the current image
            cv2.imshow('video', res)

        if len(pending) < threadn:  # fewer frames than thresds ==> get another frame
            # get frame
            ret, frame = cap.read()
            FRAME_COUNTER += 1
            t = clock()
            frame_interval.update(t - last_frame_time)
            last_frame_time = t
            if threaded_mode:
                task = pool.apply_async(process_frame, (frame.copy(), prevFrame.copy(), t))
            else:
                task = DummyTask(self.process_frame(frame, prevFrame, t))
            pending.append(task)

        # check for a keypress
        key = cv2.waitKey(1) & 0xFF

        # threaded or non threaded mode
        if key == ord(' '):
            threaded_mode = not threaded_mode
        # toggle point processes -- adjust image
        if key == ord('a'):
            ADJ_IMG = not ADJ_IMG
            if ADJ_IMG:
                cv2.createTrackbar("brightness", 'video', brightness, brightness_slider_max, on_brightness_trackbar)
                cv2.createTrackbar("contrast", 'video', contrast, contrast_slider_max, on_contrast_trackbar)
                cv2.createTrackbar("gamma", 'video', gamma, gamma_slider_max, on_gamma_trackbar)
            else:
                cv2.destroyWindow('video')
                cv2.namedWindow('video')
                cv2.imshow('video', res)
        # toggle edges
        if key == ord('e'):
            show_edges = not show_edges
            if not show_edges and not SHOW_FRAMES:
                SHOW_FRAMES = True
        # toggle frames
        if key == ord('f'):
            SHOW_FRAMES = not SHOW_FRAMES
            if not SHOW_FRAMES and not show_edges:
                SHOW_FRAMES = True
        # image difference mode
        if key == ord('d'):
            diff_frames = not diff_frames
        # display image hue band
        if key == ord('h'):
            show_hue = not show_hue
        # equalize image value band
        if key == ord('q'):
            hist_eq = not hist_eq
        # threshold the image
        if key == ord('t'):
            do_threshold = not do_threshold
            if do_threshold:
                cv2.createTrackbar("threshold", 'video', self.threshold, threshold_slider_max, on_threshold_trackbar)
            else:
                cv2.destroyWindow('video')
                cv2.namedWindow('video')
                cv2.imshow('video', res)
        # do morphological opening on thresholded image (only applied to thresholded image)
        if key == ord('o'):
            m_open = not m_open
        # write video frames
        if key == ord('v'):
            vid_frames = not vid_frames
            if vid_frames:
                print("Frames are being output to video")
            else:
                print("Frames are not being output to video")



        # ESC terminates the program
        if key == 27:
            break

# release video capture object
cap.release()
# release video output object
vid_out.release()
cv2.destroyAllWindows()
