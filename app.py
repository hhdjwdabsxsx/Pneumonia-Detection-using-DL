import os
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
MODEL_PATH = 'model/our_model.h5'
model = load_model(MODEL_PATH)

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_pneumonia(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Resize as per your model's requirement
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return "Pneumonia Detected" if predictions[0] > 0.5 else "Normal"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the POST request has the file
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            prediction = predict_pneumonia(file_path)
            return render_template('result.html', prediction=prediction, image_path=file_path)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
