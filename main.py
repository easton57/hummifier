"""
Driver for hummifier project
By: Easton Seidel
"""

import humming_ml
import standardizer
import media_reader as mr


def main():
    """ Main Function """
    # Train
    _, _, model = humming_ml.train('train_photos')

    humming_ml.save_model(model)

    humming_ml.inference(model, 'validate_photos/006.jpg')


def monitor_video():
    try:
        while True:
            pass
    except KeyboardInterrupt:
        exit()


if __name__ == "__main__":
    main()