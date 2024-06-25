#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DETECT
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import shutil
from io import BytesIO
import sys

tf.config.set_visible_devices([], 'GPU')

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class ImageClassifier:
    def __init__(self, model_path = 'model.keras'):
        self.model = load_model(model_path)

    def classify_image(self, file_buffer):     
        try:            
            img = image.load_img(BytesIO(file_buffer), target_size=(150, 150))

            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Vorhersage
            prediction = self.model.predict(img_array)[0][0]
            logging.info(f'AI CLASSIFY SUCCESS: {prediction}')
            
            return float(prediction)
        
        except Exception as e:
            logging.info(f'AI CLASSIFY: {e}')
            return None

if __name__ == '__main__':
    pass
