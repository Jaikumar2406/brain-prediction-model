from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
import io
from PIL import Image

app = Flask(__name__)

# Load your model
model = load_model('new_train3.h5')
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = image.resize((150, 150))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_image', methods=['POST'])
def upload_image():
    data = request.get_json()
    image_data = base64.b64decode(data['image'].split(',')[1])
    processed_image = preprocess_image(image_data)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]

    return jsonify({'prediction': predicted_class_name})


if __name__ == '__main__':
    app.run(debug=True)
