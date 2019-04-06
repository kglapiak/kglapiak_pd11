
import argparse
import logging
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from os import mkdir
import os
import shutil
import numpy as np


def clear_and_create(params):
    try:
        shutil.rmtree(params.out_dir, ignore_errors=True)
        mkdir(params.out_dir)
    except:
        0

def generate_images(params):
    datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

    img_1 = load_img('{0}/{1}'.format(params.data_dir,params.image_name))  # this is a PIL image
    x = np.stack([img_to_array(img_1)])  # this is a Numpy array with shape (2, 3, 150, 150)
    
    for i, batch in enumerate(datagen.flow(x, batch_size=1,
                                       save_to_dir=params.out_dir,
                                       save_prefix='cat',
                                       save_format='jpeg')):
        if i == params.image_number -1 :
            break  # otherwise the generator would loop indefinitely

def main(params):
    log = logging.getLogger('image_generator')
    clear_and_create(params)
    generate_images(params)
    log.info("Generate {0} images to: {1} !".format(params.image_number,params.out_dir))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data',
                        help="Path to directory with original images")
    
    parser.add_argument('--out_dir', type=str, default='generated',
                        help="Path to directory with out data")
     
    parser.add_argument('--image_name', type=str, default='cat.1001.jpg',
                        help="Image name")
    
    parser.add_argument('--debug', action='store_const', const=True, default=False,
                        help='Set debug logging level, otherwise info level is set.')
    
    parser.add_argument('--image_number', type=int, default=20,
                        help="Image name")
    
    params = parser.parse_args()

    # configure logger
    logger = logging.getLogger('image_generator')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()  # console handler
    ch.setLevel(logging.DEBUG if params.debug else logging.INFO)
    ch.setFormatter(logging.Formatter(fmt='%(asctime)s [%(name)s:%(levelname)s]: %(message)s',
                                      datefmt="%H:%M:%S"))
    
    logger.addHandler(ch)
    
    main(params)
