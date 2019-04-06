
import argparse
import logging
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from image_generator_01 import clear_and_create

def sub_img(params):
    
    img_list = []
    for i in os.listdir(params.data_dir):
        img = load_img( '{0}/{1}'.format(params.data_dir,i ))
        img_list.append(img_to_array(img))
        
    img_list = np.array(img_list)
    mean_img = np.load(open(params.mean_name, 'rb'))

    new_imgs =img_list - mean_img
    for i, k in enumerate(new_imgs):
        out_img = array_to_img(k)
        out_img.save("{0}/new_img_{1}.jpg".format(params.out_dir,i))

def main(params):
    log = logging.getLogger('substract_images')
    clear_and_create(params)
    sub_img(params)
    log.info("New images saved in: {0}".format(params.out_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='generated',
                        help="Path to directory with original images")
    
    parser.add_argument('--mean_name', type=str, default='mean_img.npy',
                        help="Path to directory with out data")
     
    parser.add_argument('--out_dir', type=str, default='new_images',
                        help="Path to directory with out data")
    
    parser.add_argument('--debug', action='store_const', const=True, default=False,
                        help='Set debug logging level, otherwise info level is set.')
    
    params = parser.parse_args()
    # configure logger
    logger = logging.getLogger('substract_images')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()  # console handler
    ch.setLevel(logging.DEBUG if params.debug else logging.INFO)
    ch.setFormatter(logging.Formatter(fmt='%(asctime)s [%(name)s:%(levelname)s]: %(message)s',
                                      datefmt="%H:%M:%S"))
    
    logger.addHandler(ch)
    main(params)