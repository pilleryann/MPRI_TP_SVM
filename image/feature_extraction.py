from tools import list_images
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.feature import hog


def extract_features(im):
    """ Returns a feature vector for an image patch. """

    # TODO: find other features to use
    #fd = hog(im, orientations=8, pixels_per_cell=(16, 16),
     #               cells_per_block=(1, 1), visualise=False, normalise=False)
    return im.flatten()
    #print fd.size
    #return fd


def process_image(im, border_size=5, im_size=50):
    """ Remove borders and resize """

    im = im[border_size:-border_size, border_size:-border_size]
    im = resize(im, (im_size, im_size))

    return im


def load_data(path):
    """ Return labels and features for all jpg images in path. """

    # Create a list of all files ending in .jpg
    im_list = list_images(path, '.jpg')

    # Create labels
    labels = [int(im_name.split('/')[-1][0]) for im_name in im_list]

    # Create features from the images
    # TODO: iterate over images paths
    features =[]
        # TODO: load image as a gray level image
    for im_path in im_list:
        im= np.array(Image.open(im_path).convert('L'))
        imProcessed = process_image(im)
        imagesFeatures = extract_features(imProcessed)
        features.append(imagesFeatures)
        
        
        # TODO: process the image to remove borders and resize
   
        # TODO: append extracted features to the a list
    
    # TODO: return features, and labels
    return features, labels