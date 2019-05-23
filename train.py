import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import (
    DATA_PATH,
    IMG_SIZE,
    N_DIMENSIONS,
    FEATURES_PATH,
    REDUCTER_PATH,
    TEST_PERCENTAGE,
    MODEL_PATH,
)

from dump_shuffle import load_data
from tqdm import tqdm
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from six.moves import cPickle
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import logging
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

flags = tf.flags
flags.DEFINE_string('data_path', DATA_PATH,
                    'Path of JSON file of mapped images.')
flags.DEFINE_integer('n_dimensions', N_DIMENSIONS,
                     'Number of dimensions to reduce to.')
flags.DEFINE_string('features_path', FEATURES_PATH,
                    'Path to pickle file with features.')
flags.DEFINE_string('reducter_path', REDUCTER_PATH,
                    'Path to reducter file.')
flags.DEFINE_float('test_percentage', TEST_PERCENTAGE,
                   'Percentage of data to test on.')
flags.DEFINE_string('model_path', MODEL_PATH,
                    'Path to model.')
FLAGS = flags.FLAGS


def save_pkl_file(data, file_path):
    logging.info('saving file at {}'.format(file_path))
    with open(file_path, 'wb') as pkl_file:
        cPickle.dump(data, pkl_file)


def load_pkl_file(file_path):
    logging.info('loading file at {}'.format(file_path))
    with open(file_path, 'rb') as pkl_file:
        return cPickle.load(pkl_file)


def pca(data, n_dimensions):
    logging.info('getting pca')
    assert len(data) >= n_dimensions, ('You must have same or more data than'
                                       ' n_dimensions.')

    pca = PCA(n_components=n_dimensions)

    reducted = pca.fit_transform(data)

    return reducted, pca


def create_features():
    data = load_data(FLAGS.data_path)

    logging.info('Extracting features and creating features')
    model = VGG16(weights='imagenet', include_top=False)

    session = K.get_session()
    K.set_session(session)

    img_features = []
    genders = []
    for img_path, gender_id in tqdm(data.items()):
        try:
            img = image.load_img(img_path, target_size=IMG_SIZE)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            features = model.predict(x)[0][0][0]
            img_features.append(features)
            genders.append(gender_id)

        except Exception as e:
            logging.warning('exception: {}'.format(e))

    K.clear_session()

    reduced, model = pca(img_features,
                         FLAGS.n_dimensions)

    save_pkl_file(model, FLAGS.reducter_path)
    save_pkl_file((reduced, genders), FLAGS.features_path)


def main():
    if not os.path.exists(FLAGS.features_path):
        create_features()

    X, y = load_pkl_file(FLAGS.features_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=FLAGS.test_percentage)

    logging.info('training')
    model = LogisticRegression()
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)

    print('Results on test data:\n')
    print(classification_report(y_test, predicted))

    save_pkl_file(model, FLAGS.model_path)


if __name__ == "__main__":
    main()
