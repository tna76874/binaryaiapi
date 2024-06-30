#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
binary classification API
"""
import os
from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse
from detect import *
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = Flask(__name__)
api = Api(app)

api_key = os.getenv('CNN_API_KEY', 'test')
model_path = os.getenv('CNN_MODEL_PATH', 'data/model.keras')
model_threshold = float(os.getenv('CNN_MODEL_THRESHOLD', 0.2))
blur_threshold = float(os.getenv('BLUR_THRESHOLD', 40))

classifier = ImageClassifier(model_path = model_path, threshold = model_threshold)

class ImageRating(Resource):
    def post(self):         
        try:            
            image_file = request.files.get('image')
            if not image_file:
                return {'error': 'Image file is required'}, 400
    
            auth_key = request.headers.get('Authorization')
            if not auth_key:
                return {'error': 'API key is required'}, 400
            if auth_key != api_key:
                return {'error': 'Invalid API key'}, 401
                
            loaded = FileLoader(image_file, classifier = classifier, blur_threshold=blur_threshold).load()
            
            results = loaded._get_classify_results()
            logging.info(f'RESULTS: {results}')
            
            return results, 200
        
        except Exception as e:
            return {'error': e}, 400

# Add resource to API
api.add_resource(ImageRating, '/rate')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5500)


