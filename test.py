from config import IMG_SIZE, MODEL_PATH, REDUCTER_PATH
from train import load_pkl_file

from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import argparse
import logging
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser() 
parser.add_argument("--img", type=str, default="image.jpg")
args = parser.parse_args()

flags = tf.flags
flags.DEFINE_string('img', args.img,
                    'Image path to test')
FLAGS = flags.FLAGS

def main():
    model = VGG16(weights='imagenet', include_top=False)
    session = K.get_session()
    K.set_session(session)
    img = image.load_img(FLAGS.img, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)[0][0][0]
    K.clear_session()
    reducter = load_pkl_file(FLAGS.reducter_path)
    reduced = reducter.transform([features])
    model = load_pkl_file(FLAGS.model_path)
    output = model.predict(reduced)[0]
    print('result: {}'.format(output))


if __name__ == "__main__":
    main()
