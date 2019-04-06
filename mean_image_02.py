

import argparse
import logging
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def mean_img(params):
    
    img_list = []
    for i in os.listdir(params.data_dir):
        img = load_img( '{0}/{1}'.format(params.data_dir,i ))
        img_list.append(img_to_array(img))
        
    img_list = np.array(img_list)
    mean_img = np.mean(img_list,axis=0)
    
    np.save(open('{0}.npy'.format(params.out_name), 'wb'),mean_img)
    
def main(params):
    log = logging.getLogger('mean_image')
    mean_img(params)
    log.info("Calculate mean image and save to {0}.npy !".format(params.out_name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='generated',
                        help="Path to directory with original images")
    
    parser.add_argument('--out_name', type=str, default='mean_img',
                        help="Path to directory with out data")
     
    parser.add_argument('--debug', action='store_const', const=True, default=False,
                        help='Set debug logging level, otherwise info level is set.')
    
    params = parser.parse_args()

    # configure logger
    logger = logging.getLogger('mean_image')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()  # console handler
    ch.setLevel(logging.DEBUG if params.debug else logging.INFO)
    ch.setFormatter(logging.Formatter(fmt='%(asctime)s [%(name)s:%(levelname)s]: %(message)s',
                                      datefmt="%H:%M:%S"))
    
    logger.addHandler(ch)
    
    main(params)