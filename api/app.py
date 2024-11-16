from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import base64
from PIL import Image
import random  
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np 

app = Flask(__name__)
CORS(app)


model = load_model('.\\model\\best_model.h5')
def preprocess_image(img):
    img = img.convert('RGB')
    img = img.resize((224, 224))  
    img_array = img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0 
    return img_array

def classify_image(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    return "Real" if prediction[0] > 0.5 else "Ai Generated" 

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to Ai Genearted  Image Classifier API'})

@app.route('/api/classify', methods=['POST'])
def classify():
    image_data = request.json['image']
    try:
        image_bytes = io.BytesIO(base64.b64decode(image_data))
        img = Image.open(image_bytes)
    except Exception as e:
        return jsonify({"error": "Invalid image data"}), 400
    result = classify_image(img)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
