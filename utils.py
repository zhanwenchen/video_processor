from cv2 import threshold, morphologyEx, VideoCapture, cvtColor, waitKey, \
                THRESH_BINARY, COLOR_BGR2HSV, MORPH_OPEN


# Convenience utility for OpenCV thresholding
def cv2_threshold(image_channel, my_threshold):
    return threshold(image_channel, my_threshold, 1, THRESH_BINARY)


# Convenience utility for OpenCV morphological opening
def cv2_morph_open(image_channel, structuring_element):
    return morphologyEx(image_channel, MORPH_OPEN, structuring_element, 1)


# create a video capture object
def cv2_get_video_capture_object(camera_index):
    video_capture_object = VideoCapture(camera_index)

    if video_capture_object is None or not video_capture_object.isOpened():
        raise RuntimeError('get_video_capture_object: cannot create video_capture_object with camera_index = {}'.format(camera_index))

    return video_capture_object


def get_bgr2hsv_channels(image_bgr):
    hsv = cvtColor(image_bgr, COLOR_BGR2HSV)
    return hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]  # separate the channels


def wait_key():
    return waitKey(1) & 0xFF
