import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash
import cv2
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_fyp_project_123')
csrf = CSRFProtect(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///waste_management.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'

db = SQLAlchemy(app)

# Database Models (re-defined here for simplicity)
class Log(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Load the model robustly
def load_waste_model():
    """
    Reconstructs the CNN model architecture and loads trained weights.
    A manual weight-loading strategy is used because direct h5 loading
    can be unreliable across different Keras/TensorFlow versions (specifically Keras 2 to 3).
    """
    # 1. Rebuild architecture as defined in the training notebook
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'  # Base model remains frozen with ImageNet weights
    )

    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # 2. Manually load custom trained weights for the top layers from h5 file
    model_path = 'waste_model.h5'
    if os.path.exists(model_path):
        try:
            import h5py
            with h5py.File(model_path, 'r') as f:
                # Target the specific weight paths found in the H5 structure
                d1_k = f['model_weights/dense/sequential/dense/kernel'][:]
                d1_b = f['model_weights/dense/sequential/dense/bias'][:]
                model.layers[2].set_weights([d1_k, d1_b])

                d2_k = f['model_weights/dense_1/sequential/dense_1/kernel'][:]
                d2_b = f['model_weights/dense_1/sequential/dense_1/bias'][:]
                model.layers[3].set_weights([d2_k, d2_b])
            print("Successfully loaded trained top-layer weights.")
        except Exception as e:
            print(f"Warning: Could not load custom weights ({e}). Using default initialization.")
    else:
        print(f"Warning: {model_path} not found. Running with default weights.")

    return model

model = load_waste_model()

# Create database tables
with app.app_context():
    db.create_all()
    # Create a default admin user if it doesn't exist
    from werkzeug.security import generate_password_hash
    if not User.query.filter_by(username='admin').first():
        hashed_password = generate_password_hash('admin123')
        admin_user = User(username='admin', password=hashed_password)
        db.session.add(admin_user)
        db.session.commit()

# Mapping classes (Assuming alphabetical order from TrashNet/Keras flow_from_directory)
CLASS_LABELS = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Organic Material']

def predict_label(img_path):
    # Using OpenCV as suggested in the FYP Proposal
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    # Threshold to handle non-waste or unrecognized objects (FYP Proposal: Edge Cases)
    CONFIDENCE_THRESHOLD = 0.5
    if confidence < CONFIDENCE_THRESHOLD:
        return "Unrecognized", confidence

    return CLASS_LABELS[class_idx], confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classifier', methods=['GET', 'POST'])
def classifier():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            label, confidence = predict_label(filepath)

            # Log to database
            new_log = Log(filename=filename, prediction=label, confidence=confidence)
            db.session.add(new_log)
            db.session.commit()

            return render_template('classifier.html', filename=filename, label=label, confidence=f"{confidence*100:.2f}%")

    return render_template('classifier.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid credentials')

    if session.get('logged_in'):
        return redirect(url_for('admin_dashboard'))

    return render_template('login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('admin'))

    logs = Log.query.order_by(Log.timestamp.desc()).all()
    return render_template('admin.html', logs=logs, classes=CLASS_LABELS)

@app.route('/admin/upload_dataset', methods=['POST'])
def upload_dataset():
    if not session.get('logged_in'):
        return redirect(url_for('admin'))

    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('admin_dashboard'))

    file = request.files['file']
    category = request.form.get('category')

    if file.filename == '' or not category:
        flash('No selected file or category')
        return redirect(url_for('admin_dashboard'))

    if file and category in CLASS_LABELS:
        filename = secure_filename(file.filename)
        # Ensure category folder exists
        category_path = os.path.join('static/dataset', category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

        file.save(os.path.join(category_path, filename))
        flash(f'Successfully uploaded {filename} to {category}')

    return redirect(url_for('admin_dashboard'))

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
