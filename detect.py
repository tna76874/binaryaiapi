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
from PIL import Image
import cv2
import requests
from pdf2image import convert_from_bytes
import magic
import mimetypes
from functools import wraps
import hashlib
import logging

tf.config.set_visible_devices([], 'GPU')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def none_on_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # print(f'Error: {e}')
            return None
    return wrapper

class BrightnessCheck:
    def __init__(self, file_buffer, min_brightness=100, max_brightness=252):
        self.file_buffer = file_buffer
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.brightness = self._calculate_brightness()

    def _calculate_brightness(self):
        image = Image.open(BytesIO(self.file_buffer))
        greyscale_image = image.convert('L')
        np_image = np.array(greyscale_image)
        brightness = np.mean(np_image)
        return brightness

    def check(self):
        if self.min_brightness <= self.brightness <= self.max_brightness:
            return True
        else:
            return False

class AspectRatioCheck:
    def __init__(self, file_buffer, tolerance=0.5):
        self.file_buffer = file_buffer
        self.ratio = self._aspect_ratio()
        self.normed_ratio = None
        self.tolerance = tolerance

    @none_on_exception
    def _aspect_ratio(self):
        image = Image.open(BytesIO(self.file_buffer))
        width, height = image.size
        aspect_ratio = height / width
        return aspect_ratio
    
    @none_on_exception
    def check(self):
        if self.ratio==None:
            return None
        self.normed_ratio = self.ratio/np.sqrt(2)
        if abs(self.normed_ratio - 1) <= self.tolerance:
            return True
        else:
            return False

class FileLoader:
    def __init__(self, file_input, max_file_size = 15 * 1024 * 1024, filename = None, classifier = None, blur_threshold = 40, ratio_tolerance=0.1):
        self.attributes =   {
                            'filename' : filename,
                            }
        self.pages = list()
        self.max_file_size = max_file_size
        self.load_buffer(file_input)
        
        self.classifier = classifier
        self.blur_threshold = blur_threshold
        self.ratio_tolerance = ratio_tolerance

    def __del__(self):
        self._close_file()
        
    def load(self):
        self.get_mimetype()
        self.calc_checksum()
        self.check_filesize()
        
        self._close_file()
        self._generate_pages()
        self._classify_pages()
        self._check_aspect_ratio()
        self._check_brightness()
        
        return self
    
    @none_on_exception
    def _get_classify_results(self):
        if len(self.pages) == 0:
            return {'brightness': None, 'ratio': None, 'cnn': None, 'blur': None, 'pass' : False}

        cnn_status = [page.get('cnn', {}).get('status') for page in self.pages]
        blur_status = [page.get('blur', {}).get('status') for page in self.pages]
        ratio_status = [page.get('ratio', {}).get('status') for page in self.pages]
        brightness_status = [page.get('brightness', {}).get('status') for page in self.pages]
        
        # CNN Mean Prediction
        cnn_prediction_values = np.array([page.get('cnn', {}).get('prediction') for page in self.pages if page.get('cnn', {}).get('prediction') is not None], dtype=float)
        cnn_prediction_mean = np.mean(cnn_prediction_values) if cnn_prediction_values.size > 0 else None
                
        # Blur Mean Variance
        blur_variance_values = np.array([page.get('blur', {}).get('variance') for page in self.pages if page.get('blur', {}).get('variance') is not None], dtype=float)
        blur_variance_mean = np.mean(blur_variance_values) if blur_variance_values.size > 0 else None
        
        # Brightness Mean
        brightness_values = np.array([page.get('brightness', {}).get('brightness') for page in self.pages if page.get('brightness', {}).get('brightness') is not None], dtype=float)
        brightness_mean = np.mean(brightness_values) if brightness_values.size > 0 else None
        
        cnn_pass = all(cnn_value == True for cnn_value in cnn_status)
        blur_pass = all(blur_value == False for blur_value in blur_status)
        ratio_pass = all(ratio_value == True for ratio_value in ratio_status)
        brightness_pass = all(brightness_value == True for brightness_value in brightness_status)
        
        all_pass = cnn_pass and blur_pass and ratio_pass and brightness_pass

        results = {'brightness' : brightness_pass, 'ratio' : ratio_pass, 'cnn': cnn_pass, 'blur': blur_pass, 'pass' : all_pass, 'values' : {'cnn' : cnn_prediction_mean, 'blur' : blur_variance_mean, 'brightness' : brightness_mean}}
        
        return results
    
    @none_on_exception
    def _classify_pages(self):
        for idx, page in enumerate(self.pages):
            inspect = page.get('buffer')
            if self.classifier != None:
                cnn = self.classifier.classify_image(inspect)
                self.pages[idx].update({'cnn' : cnn})

            self.pages[idx].update({'blur' : DetectBlur(threshold = self.blur_threshold).detect_blur(inspect)})
            

    @none_on_exception
    def _check_aspect_ratio(self):
        for idx, page in enumerate(self.pages):
            ratio = AspectRatioCheck(page['buffer'], tolerance=self.ratio_tolerance)
            ratio_status = ratio.check()
            self.pages[idx].update({'ratio' : {'status' : ratio_status, 'normed_ratio' : ratio.normed_ratio}})
            
    @none_on_exception
    def _check_brightness(self):
        for idx, page in enumerate(self.pages):
            brightness = BrightnessCheck(page['buffer'])
            brightness_status = brightness.check()
            self.pages[idx].update({'brightness' : {'status' : brightness_status, 'brightness' : brightness.brightness}})
    
    @none_on_exception
    def _generate_pages_from_pdf(self):
        images = convert_from_bytes(self.buffer, fmt="jpeg")
        for image in images:
            byte_buffer = BytesIO()
            image.save(byte_buffer, format='JPEG')
            image_bytes = byte_buffer.getvalue()
            byte_buffer.close()
            self.pages.append(
                                {'buffer' : image_bytes}
                              )
    
    @none_on_exception
    def _generate_pages(self):
        if self.attributes.get('is_image') == True:
            self.pages.append(
                                {'buffer' : self.buffer}
                              )
            
        elif self.attributes.get('is_pdf') == True:
            self._generate_pages_from_pdf()

    def max_size_mb(self):
        return self.max_file_size / (1024 * 1024)
    
    @none_on_exception
    def _check_if_is_image(self):
        image_mimetypes = [mimetype for mimetype in mimetypes.types_map.values() if mimetype.startswith('image')] + ['image/heic']
        self.attributes.update({'is_image': self.attributes.get('mimetype') in image_mimetypes})
                                
    @none_on_exception
    def _check_if_is_pdf(self):
        pdf_mimetypes = [mime_type for mime_type in mimetypes.types_map.values() if 'pdf' in mime_type]
        self.attributes.update({'is_pdf': self.attributes.get('mimetype') in pdf_mimetypes})
        
    @none_on_exception
    def _check_if_filetype_is_accepted(self):
        accept_file_mimetype = self.attributes.get('is_pdf', False) or self.attributes.get('is_image', False)
        self.attributes.update({'accept_mimetype': accept_file_mimetype})

    @none_on_exception
    def check_filesize(self):
        self.attributes.update({'size': len(self.buffer) <= self.max_file_size})

    @none_on_exception
    def calc_checksum(self):
        sha256_hash = hashlib.sha256()
        sha256_hash.update(self.buffer)
        self.sha256_hash = sha256_hash.hexdigest()
        self.attributes.update({'sha256_hash':self.sha256_hash})
    
    @none_on_exception
    def get_mimetype(self):
        mime = magic.Magic(mime=True)
        self.mime_type = mime.from_buffer(self.buffer)  
        self.attributes.update({'mimetype':self.mime_type})
        
        self._check_if_is_image()
        self._check_if_is_pdf()
        self._check_if_filetype_is_accepted()
        
    @none_on_exception
    def _open_file(self, file_input):
        self.file = open(file_input, 'rb')
        
    @none_on_exception
    def _read_file(self):
        file_read = self.file.read()
        self.file.seek(0)
        return file_read

    @none_on_exception
    def _close_file(self):
        self.file.close()
    
    @none_on_exception
    def load_buffer(self, file_input):
        if isinstance(file_input, str):
            if not os.path.isfile(file_input):
                raise FileNotFoundError(f'Die Datei {file_input} wurde nicht gefunden.')
            self._open_file(file_input)
        else:
            self.file = file_input
        
        self.buffer = self._read_file()

class DetectBlur:
    def __init__(self, threshold=40):
        self.threshold = threshold

    def detect_blur(self, file_buffer):
        try:           
            # Konvertiere die Binärdaten in ein numpy-Array
            img_array = np.frombuffer(file_buffer, dtype=np.uint8)
            
            # Lade das Bild mit OpenCV aus dem numpy-Array
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
            # Apply binary thresholding for bright spot detection
            _, binary_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
            # Apply Laplacian filter for edge detection
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
            # Calculate maximum intensity and variance
            _, max_val, _, _ = cv2.minMaxLoc(gray)
            laplacian_variance = laplacian.var()
    
            # Initialize result variables
            blur_text = f"Not Blurry ({laplacian_variance})"
    
            # Check blur condition based on variance of Laplacian image
            is_blurred = laplacian_variance < self.threshold
            if is_blurred:
                blur_text = f"Blurry ({laplacian_variance})"
    
            return {'status' : is_blurred, 'variance': laplacian_variance}
        except:
            return {'status' : False, 'variance': None}

class ImageClassifier:
    def __init__(self, model_path = 'model.keras', threshold = 0.20):
        self.model = load_model(model_path)
        self.threshold = threshold

    def classify_image(self, file_buffer):     
        try:            
            img = image.load_img(BytesIO(file_buffer), target_size=(150, 150))

            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Vorhersage
            prediction = self.model.predict(img_array)[0][0]
            
            prediction = float(prediction)
            status = prediction <= self.threshold
            
            return {'prediction' : prediction, 'status' : status}
        
        except Exception as e:
            logging.info(f'AI CLASSIFY: {e}')
            return None

if __name__ == '__main__':
    pass
