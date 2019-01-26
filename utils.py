from cv2 import threshold, THRESH_BINARY, morphologyEx, MORPH_OPEN


# Convenience utility for OpenCV thresholding
def cv2_threshold(image_channel, threshold):
    return threshold(image_channel, threshold, 1, THRESH_BINARY)


# Convenience utility for OpenCV morphological opening
def cv2_morph_open(image_channel, structuring_element):
    return morphologyEx(image_channel, MORPH_OPEN, structuring_element, 1)
