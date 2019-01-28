# main program
# Standard (core library) imports.

# Non-standard (installeed library) imports.
import numpy as np

# Relative (local) imports
from video_processor import VideoProcessor

# Global variables.
CAMERA_INDEX = 0


def main():
    video_processor = VideoProcessor()
    video_processor.run()


if __name__ == '__main__':
    # print in the program shell window the text at the beginning of the file
    main()
