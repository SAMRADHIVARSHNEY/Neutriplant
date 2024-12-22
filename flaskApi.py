# from flask import Flask, render_template, request, redirect, url_for
# import tensorflow as tf
# import numpy as np
# from werkzeug.utils import secure_filename
# import os
# from PIL import Image
# import io

# # Flask setup
# app = Flask(__name__)

# # Model loading
# model = tf.keras.models.load_model("trained_plant_disease_model.keras")

# # Image upload path
# UPLOAD_FOLDER = 'static/uploads'  # Make sure it's under static for Flask to serve the files
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Make sure the uploads directory exists
# if not os.path.exists(app.config['UPLOAD_FOLDER']):
#     os.makedirs(app.config['UPLOAD_FOLDER'])

# # Helper function to check file extension
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # Helper function to process the image and make prediction
# def model_prediction(image_data):
#     image = Image.open(io.BytesIO(image_data))
#     image = image.resize((128, 128))
#     image = np.array(image)
#     image = np.expand_dims(image, axis=0)  # Convert single image to batch
#     predictions = model.predict(image)
#     return np.argmax(predictions)  # Return index of max element

# # Disease class names
# class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
#                 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
#                 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
#                 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
#                 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
#                 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
#                 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
#                 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
#                 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
#                 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
#                 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
#                 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
#                 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
#                 'Tomato___healthy']

# # Route for the homepage
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route for handling image upload and prediction
# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
    
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         # Read image for prediction
#         with open(filepath, 'rb') as f:
#             image_data = f.read()
#             result_index = model_prediction(image_data)
#             predicted_class = class_names[result_index]
        
#         # Return result page with prediction
#         return render_template('result.html', prediction=predicted_class, image_path=filepath)
    
#     return redirect(url_for('index'))

# # Route for handling webcam capture (if using webcam capture method)
# @app.route('/capture', methods=['POST'])
# def capture_image():
#     if 'image' not in request.files:
#         return redirect(request.url)
#     image = request.files['image']
    
#     if image and allowed_file(image.filename):
#         image_data = image.read()
#         result_index = model_prediction(image_data)
#         predicted_class = class_names[result_index]
        
#         # Save the image for display
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.jpg')
#         image.save(filepath)
        
#         # Return result page with prediction
#         return render_template('result.html', prediction=predicted_class, image_path='uploads/captured_image.jpg')
    
#     return redirect(url_for('index'))

# if __name__ == '__main__':
#     app.run(debug=True)




# from flask import Flask, request
# import os

# app = Flask(__name__)

# UPLOAD_FOLDER = './uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'image' not in request.files:
#         return 'No image part in request', 400

#     file = request.files['image']
#     if file.filename == '':
#         return 'No selected file', 400

#     # Save the file
#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(file_path)

#     return 'Image uploaded successfully', 200

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0')



# from flask import Flask, request, jsonify
# import os
# from keras.models import load_model
# from PIL import Image
# import numpy as np

# # Define your Flask app
# app = Flask(__name__)

# # Path to the uploads folder
# UPLOAD_FOLDER = './uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# # Load the trained model
# model = load_model('trained_plant_disease_model.keras')

# # Define the class names
# class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
#                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
#                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
#                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
#                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
#                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
#                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
#                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
#                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
#                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
#                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
#                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
#                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
#                'Tomato___healthy']

# # Image size expected by the model
# IMAGE_SIZE = (128, 128)  # Use 128x128 as per your model's expected input size

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image part in request'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     # Save the file
#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(file_path)

#     # Load and preprocess the image for prediction
#     image = Image.open(file_path)
#     image = image.convert('RGB')  # Convert image to RGB (removes alpha channel if present)
#     image = image.resize(IMAGE_SIZE)  # Resize the image to match model input size
#     image_array = np.array(image) / 255.0  # Normalize the image
#     image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

#     # Predict the class of the image
#     predictions = model.predict(image_array)
#     predicted_class = class_names[np.argmax(predictions)]

#     # Return the prediction result
#     return jsonify({'predicted_class': predicted_class}), 200

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0')




from flask import Flask, request, jsonify
import os
from keras.models import load_model
from PIL import Image
import numpy as np

# Define your Flask app
app = Flask(__name__)

# Path to the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model = load_model('trained_plant_disease_model.keras')

# Define the class names as per your model
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Image size expected by the model
IMAGE_SIZE = (128, 128)  # Use 128x128 as per your model's expected input size

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Load and preprocess the image for prediction
    try:
        image = Image.open(file_path)
        image = image.convert('RGB')  # Convert image to RGB (removes alpha channel if present)
        image = image.resize(IMAGE_SIZE)  # Resize the image to match model input size
        image_array = np.array(image) / 255.0  # Normalize the image
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict the class of the image
        predictions = model.predict(image_array)
        predicted_class = class_names[np.argmax(predictions)]

        # Return the prediction result
        return jsonify({'predicted_class': predicted_class}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
