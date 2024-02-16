"""
Driver for hummifier project
By: Easton Seidel
"""

import humming_ml
import standardizer
import media_reader as mr


def main():
    """ Main Function """
    # Call to read images
    ml_data = mr.read_images('photos')


def monitor_video():
    try:
        while True:
            pass
    except KeyboardInterrupt:
        exit()


if __name__ == "__main__":
    main()