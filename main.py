import os
import tensorflow as tf
from numpy import expand_dims
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import load_img, img_to_array


if __name__=='__main__':
    tf.keras.backend.clear_session()
    input_shape = (224, 224, 3)
    mobilenet = MobileNet(
        input_shape=input_shape,
        alpha=1.0,
        depth_multiplier=1,
        dropout=0.001,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        pooling=None,
        classes=1000, 
        classifier_activation='softmax'
    )
    base_path = os.getcwd()
    path_images = os.path.join(base_path, 'images')
    images = os.listdir(path_images)
    for image in images:
        path = os.path.join(path_images, image)
        img = load_img(path, target_size=input_shape)
        img_array = img_to_array(img)
        img_array_expanded_dims = expand_dims(img_array, axis=0)
        img_processed = tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
        prediction = mobilenet.predict(img_processed)
        results = imagenet_utils.decode_predictions(prediction)
        print(results)
