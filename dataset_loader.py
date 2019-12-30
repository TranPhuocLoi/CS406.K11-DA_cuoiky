# Import necessary packages
import numpy as np
import cv2 as cv
import os

class DatasetLoader:

    def load(self, image_paths, verbose=-1):

        # Initialize the list of images and labels
        data = []
        labels = []

        # Loop over input paths to read the data
        for (i, path) in enumerate(image_paths):
            # Load images
            # Assuming path in following format:
            # /path/to/dataset/{class}/{image-name}.jpg
            image = cv.imread(path)
            label = path.split(os.path.sep)[-2]

            # Resize image
            image = cv.resize(image, (32, 32))

            # Push into data list
            data.append(image)
            labels.append(label)

            # Show update
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(image_paths)))

        # Return a tuple of data and labels
        return (np.array(data), np.array(labels))