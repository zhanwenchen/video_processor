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
from collections import deque
from multiprocessing.pool import ThreadPool

# Non-standard imports. They have been installed by conda or pip.
import numpy as np
# import cv2
from cv2 import Canny, add, cvtColor, applyColorMap, merge, absdiff, \
                getStructuringElement, equalizeHist, createTrackbar, \
                imshow, destroyWindow, namedWindow, destroyAllWindows, \
                getNumberOfCPUs, VideoWriter, VideoWriter_fourcc, \
                COLOR_GRAY2RGB, COLOR_BGR2HSV, COLOR_HSV2BGR, COLORMAP_HSV, MORPH_ELLIPSE, \
                CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT

# Relative (local) imports. They are defined in your project folder(s).
# import video
from common import clock, draw_str, StatValue
from utils import cv2_threshold, cv2_morph_open, cv2_get_video_capture_object, \
                  get_bgr2hsv_channels, wait_key


# Global variables (constants). They should be SCREAMING_SNAKE_CASE and read-only.
CONTRAST_SLIDER_MAX = 255
BRIGHTNESS_SLIDER_MAX = 255
GAMMA_SLIDER_MAX = 255
THRESHOLD_SLIDER_MAX = 255
WINDOW_NAME = 'VideoProcessor' # The name on top of the video player GUI
# By default, use the 0th camera (e.g. on a laptop, it's the integrated webcam).
# NOTE that when your laptop connects to a monitor with its own integrated
# webcam, the external monitor's camera is now incremented to (probably 1)
# whether your laptop lid is closed.
DEFAULT_CAMERA_INDEX = 0
VIDEO_OUTPUT_FNAME = 'vid_out.avi'
ESC_KEY = 27

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
        frame_counter (int): The number of frames recorded so far.
        showing_frame (bool): Default True. Whether to show the raw or
                              transformed frame. Alternatively, one can set this
                              to False and only see edges (if showing_edges is
                              True).
        diff_frames (bool):
        showing_edges (bool): Default False. Whether to show (or overlay) edges.
    show_hue:
    do_threshold:
    adj_img:
    adj_gam:
    m_open:
    hist_eq:
    writing_frame_to_video:
    contrast:
    brightness:
    gamma:
    gamma_slider_max:
    threshold:
    raw_frame ():
    prev_frame ():
    computed_frame ():
    threaded_mode (bool): Default True
    '''
    def __init__(self, camera_index=DEFAULT_CAMERA_INDEX):
        self.frame_counter = 0
        self.showing_frame = True
        self.showing_diff_frames = False
        self.showing_edges = False
        self.showing_hue = False
        self.doing_threshold = False
        self.threshold = 128
        self.doing_gamma_and_contrast_adjust = False
        self.gamma = 128
        self.doing_morph_open = False
        self.doing_hist_eq = False
        self.writing_frame_to_video = False
        self.contrast = 128
        self.brightness = 128

        self.threaded_mode = True

        # Initialize video capture object and capture the initial frame.
        self.video_capture_obj = cv2_get_video_capture_object(camera_index)

        self.computed_frame = None

        self.current_frame_time = None


    def get_raw_frame(self):
        _, frame = self.video_capture_obj.read()
        if frame is None:
            raise RuntimeError('{}.get_raw_frame: frame is None'.format(__name__))

        self.frame_counter += 1
        self.current_frame_time = clock()
        return frame

    def increment_frame_counter(self):
        self.frame_counter += 1


    def apply_gamma_and_contrast_adjust(self):
        '''
        '''
        # shift value to get actual brightness offset
        new_brightness = self.brightness - 128
        # compute the contrast value from the trackbar setting
        if self.contrast > 127:
            new_contrast = 1+(5*(self.contrast - 128)/128)
        else:
            new_contrast = 1/(1+(5*(128 - self.contrast)/128))
        # adjust brightness and contrast
        self.computed_frame = ((np.float64(self.computed_frame)-128) * new_contrast) + 128 + new_brightness
        # compute the gamma value from the trackbar setting
        if self.gamma > 127:
            new_gamma = 1+(2*(self.gamma - 128)/128)
        else:
            new_gamma = 1/(1+(2*(128 - self.gamma)/128))
        # apply the gamma function
        self.computed_frame = 255 * ((self.computed_frame / 255) ** (1 / new_gamma))
        # then convert the result back to uint8 after clipping at 0 and 255
        np.clip(self.computed_frame, 0, 255, out=self.computed_frame) # inplace with 'out' is a bit faster.
        self.computed_frame = np.uint8(self.computed_frame)

        createTrackbar('brightness', WINDOW_NAME, self.brightness, BRIGHTNESS_SLIDER_MAX, self.set_brightness)
        createTrackbar('contrast', WINDOW_NAME, self.contrast, CONTRAST_SLIDER_MAX, self.set_contrast)
        createTrackbar('gamma', WINDOW_NAME, self.gamma, GAMMA_SLIDER_MAX, self.set_gamma)


    def apply_hist_eq(self):
        '''
        '''
        hue, sat, val = get_bgr2hsv_channels(self.computed_frame)
        val = equalizeHist(val)
        hsv = merge((hue, sat, val))
        hist_eqed_frame = cvtColor(hsv, COLOR_HSV2BGR)
        self.computed_frame = hist_eqed_frame


    def get_and_show_edges_from_computed_frame(self):
        '''
        Judge whether to show frame here, because otherwise we lose the frame
        by setting computed_frame to the edges.
        '''
        edges = Canny(self.computed_frame, 100, 200)
        edges_rgb = cvtColor(edges, COLOR_GRAY2RGB, edges)
        if self.showing_frame:
            self.computed_frame = add(self.computed_frame, edges_rgb)
        if not self.showing_frame:
            self.computed_frame = edges_rgb


    def create_and_apply_hue_from_computed_frame(self):
        '''
        Does not modify new_frame.
        '''
        # convert image to HSV
        hue, _, _ = get_bgr2hsv_channels(self.computed_frame)
        # the maximum hue is 170, so scale it into [0,255]
        h32 = np.float32(hue) * 255 / 170
        np.clip(h32, 0, 255, out=h32)
        # sch = np.uint8(h32)  # clip at 255 and convert back to uint8
        # apply the opencv builtin hue colormap
        # hue = applyColorMap(sch, COLORMAP_HSV)
        # hue = applyColorMap(sch, COLORMAP_HSV)
        self.computed_frame = applyColorMap(np.uint8(h32), COLORMAP_HSV)
        # self.computed_frame = hue
        # return hue


    def create_and_apply_threshold_mask(self):
        B, G, R = self.computed_frame[:, :, 0], self.computed_frame[:, :, 1], self.computed_frame[:, :, 2]
        _, B_thresholded = cv2_threshold(B, self.threshold)
        _, G_thresholded = cv2_threshold(G, self.threshold)
        _, R_thresholded = cv2_threshold(R, self.threshold)
        if self.doing_morph_open:
            # create structuring element for morph ops
            structuring_element = getStructuringElement(MORPH_ELLIPSE, (5, 5))
            B_thresholded = cv2_morph_open(B_thresholded, structuring_element)
            G_thresholded = cv2_morph_open(G_thresholded, structuring_element)
            R_thresholded = cv2_morph_open(R_thresholded, structuring_element)
        threshold_mask = merge((B_thresholded, G_thresholded, R_thresholded))

        # Apply threshold mask
        self.computed_frame *= threshold_mask # REVIEW: Is *= behaving correctly?


    # this routine is run each time a new video frame is captured
    def process_frame(self):
        '''

        Note:
            The order of operations is optional.
        Args:
        Returns:
            None
        '''
        # A placeholder for accumulating operations for the new frame.
        # new_frame = np.zeros_like(self.raw_frame, dtype=np.uint8)
        prev_frame = self.computed_frame.copy()

        # Reset computed_frame to raw frame
        self.computed_frame = self.get_raw_frame()

        if self.doing_gamma_and_contrast_adjust:
            self.apply_gamma_and_contrast_adjust()

        if self.doing_hist_eq:
            self.apply_hist_eq()

        if self.doing_threshold:
            # create and apply threshold mask
            self.create_and_apply_threshold_mask()

        if self.showing_edges:
            self.get_and_show_edges_from_computed_frame()

        if self.showing_hue:
            self.create_and_apply_hue_from_computed_frame()

        if self.showing_diff_frames:
            self.computed_frame = absdiff(self.computed_frame, prev_frame)


    def remove_trackbars(self):
        '''When don't nedd trackbars, destroy current window and redraw it'''
        destroyWindow(WINDOW_NAME)
        namedWindow(WINDOW_NAME)
        imshow(WINDOW_NAME, self.computed_frame)


    def set_brightness(self, val):
        self.brightness = val


    def set_contrast(self, val):
        self.contrast = val


    def set_gamma(self, val):
        self.gamma = val


    def set_threshold(self, val):
        self.threshold = val


    def toggle_doing_gamma_and_contrast_adjust(self):
        print('{}.toggle_doing_gamma_and_contrast_adjust: hit'.format(__name__))
        self.doing_gamma_and_contrast_adjust = not self.doing_gamma_and_contrast_adjust
        if self.doing_gamma_and_contrast_adjust:
            createTrackbar('gamma', WINDOW_NAME, self.gamma, GAMMA_SLIDER_MAX, self.set_gamma)
            createTrackbar('brightness', WINDOW_NAME, self.brightness, BRIGHTNESS_SLIDER_MAX, self.set_brightness)
            createTrackbar('contrast', WINDOW_NAME, self.contrast, CONTRAST_SLIDER_MAX, self.set_contrast)
        else:
            self.remove_trackbars()


    def show_frames_if_not_showing_frames_and_edges(self):
        '''
        Assuming we don't what to show nothing, when not showing even frame
        or edges, we treat that as a mistake and just set showing_frame to
        true.
        '''
        if not self.showing_frame and not self.showing_edges:
            self.showing_frame = True


    def toggle_showing_edges(self):
        self.showing_edges = not self.showing_edges
        self.show_frames_if_not_showing_frames_and_edges()


    def toggle_showing_frame(self):
        self.showing_frame = not self.showing_frame
        self.show_frames_if_not_showing_frames_and_edges()


    def toggle_doing_threshold(self):
        self.doing_threshold = not self.doing_threshold
        if self.doing_threshold:
            createTrackbar('threshold', WINDOW_NAME, self.threshold, THRESHOLD_SLIDER_MAX, self.set_threshold)
        else:
            self.remove_trackbars()


    def toggle_writing_frame_to_video(self):
        self.writing_frame_to_video = not self.writing_frame_to_video
        if self.writing_frame_to_video:
            print('{}.toggle_writing_frame_to_video: Frames are being output to video'.format(__name__))
        else:
            print('{}.toggle_writing_frame_to_video: Frames are not being output to video'.format(__name__))


    def run(self):
        print(__doc__)
        wait_key()
        self.computed_frame = self.get_raw_frame()
        namedWindow(WINDOW_NAME)

        # Create video of Frame sequence -- define the codec and create VideoWriter object
        fourcc = VideoWriter_fourcc(*'XVID')
        cols = int(self.video_capture_obj.get(CAP_PROP_FRAME_WIDTH))
        rows = int(self.video_capture_obj.get(CAP_PROP_FRAME_HEIGHT))
        vid_out = VideoWriter(VIDEO_OUTPUT_FNAME, fourcc, 20.0, (cols, rows))

        # Set up multiprocessing
        num_threads = getNumberOfCPUs()
        pool = ThreadPool(processes=num_threads)
        pending = deque()

        # initialize time variables
        latency = StatValue()
        frame_interval = StatValue()

        # main program loop
        while True:
            while pending and pending[0].ready():  # there are frames in the queue
                pending.popleft().get()
                latency.update(clock() - self.current_frame_time)
                # plot info on threading and timing on the current image
                # comment out the next 3 lines to skip the plotting
                draw_str(self.computed_frame, (20, 20), "threaded      :  " + str(self.threaded_mode))
                draw_str(self.computed_frame, (20, 40), "latency        :  %.1f ms" % (latency.value * 1000))
                draw_str(self.computed_frame, (20, 60), "frame interval :  %.1f ms" % (frame_interval.value * 1000))
                # write output video frame
                if self.writing_frame_to_video:
                    vid_out.write(self.computed_frame)
                # show the current image
                imshow(WINDOW_NAME, self.computed_frame)

            if len(pending) < num_threads:  # fewer frames than thresds ==> get another frame
                # get frame
                if self.threaded_mode:
                    task = pool.apply_async(self.process_frame)
                else:
                    task = DummyTask(self.process_frame())
                frame_interval.update(clock() - self.current_frame_time)
                pending.append(task)

            # check for a keypress
            key = wait_key()

            # threaded or non threaded mode
            if key == ord(' '):
                self.threaded_mode = not self.threaded_mode
            # toggle doing_gamma_and_contrast_adjust
            if key == ord('a'):
                self.toggle_doing_gamma_and_contrast_adjust()
            # toggle edges
            if key == ord('e'):
                self.toggle_showing_edges()
            # toggle frame
            if key == ord('f'):
                self.toggle_showing_frame()
            # image difference mode
            if key == ord('d'):
                self.showing_diff_frames = not self.showing_diff_frames
            # display image hue band
            if key == ord('h'):
                self.showing_hue = not self.showing_hue
            # equalize image value band
            if key == ord('q'):
                self.doing_hist_eq = not self.doing_hist_eq
            # threshold the image
            if key == ord('t'):
                self.toggle_doing_threshold()
            # do morphological opening on thresholded image (only applied to thresholded image)
            if key == ord('o'):
                self.doing_morph_open = not self.doing_morph_open
            # write video frames
            if key == ord('v'):
                self.toggle_writing_frame_to_video()

            # ESC terminates the program
            if key == ESC_KEY:
                break

        # release video capture object
        self.video_capture_obj.release()
        # release video output object
        vid_out.release()
        destroyAllWindows()
