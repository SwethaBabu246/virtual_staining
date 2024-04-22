import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import io
import PIL.Image

app = Flask(__name__)

# Define paths
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if file has allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Load the trained Pix2Pix GAN model
pix2pix_model = tf.keras.models.load_model("models/pix2pix_model.h5")

# Load the trained cancer detection model
cancer_detection_model = tf.keras.models.load_model("models/cancer_detection_model.h5")

# Function to preprocess a single image
def preprocess_image(image):
    img = tf.image.decode_image(image.read(), channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (256, 256))  # Resize to model's input shape
    img = img[tf.newaxis, :]
    return img

# Function to generate random predictions
def generate_random_predictions(num_samples):
    return np.random.uniform(low=0, high=100, size=(num_samples,))

# Route for uploading image and processing with Pix2Pix and cancer detection models
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # Check if the file is empty
        if file.filename == '':
            return redirect(request.url)
        # Check if the file has allowed extension
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Preprocess the uploaded image
            with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as f:
                img = preprocess_image(f)
            # Generate stained image using Pix2Pix model
            generated_stained_img = pix2pix_model(img)
            # Convert generated stained image tensor to PIL image
            generated_image = tensor_to_image(generated_stained_img)
            # Save generated image with prefixed filename
            generated_filename = 'generated_' + filename
            generated_image.save(os.path.join(app.config['UPLOAD_FOLDER'], generated_filename))
            # Predict cancer percentage using the cancer detection model
            cancer_percentage = cancer_detection_model.predict(generated_stained_img)[0][0]
            # Generate random predictions
            random_predictions = generate_random_predictions(5)  # Change 5 to the desired number of random predictions
            return render_template('result.html', original_image=filename, generated_image=generated_filename, cancer_percentage=cancer_percentage, random_predictions=random_predictions)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
