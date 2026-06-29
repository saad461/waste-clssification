import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename
import cv2
from datetime import datetime
import csv
import io
from flask import make_response
from dotenv import load_dotenv
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Professional Security: Use an environment variable or a secure fallback key
app.secret_key = os.environ.get('SECRET_KEY', 'waste-classification-fyp-secure-key-78921')
csrf = CSRFProtect(app)

# Database Configuration (MySQL Migration)
# Allow overriding URI for testing
if os.environ.get('SQLALCHEMY_DATABASE_URI'):
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('SQLALCHEMY_DATABASE_URI')
else:
    DB_USER = os.environ.get('MYSQL_USER', 'root')
    DB_PASSWORD = os.environ.get('MYSQL_PASSWORD', '')
    DB_HOST = os.environ.get('MYSQL_HOST', 'localhost')
    DB_PORT = os.environ.get('MYSQL_PORT', '3306')
    DB_NAME = os.environ.get('MYSQL_DB', 'waste_classification')

    app.config['SQLALCHEMY_DATABASE_URI'] = (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(10), default='user', nullable=False)  # 'admin' or 'user'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Log(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    feedback = db.Column(db.String(20), nullable=True) # 'Correct', 'Incorrect', or None
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

# Load the model robustly
def load_waste_model():
    model_path = 'waste_model.h5'
    if not os.path.exists(model_path):
        print("ERROR: waste_model.h5 not found.")
        return None

    # Strategy 1: Full model load
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully (full load).")
        return model
    except Exception as e:
        print(f"Full load failed: {e}. Trying weight reconstruction...")

    # Strategy 2: Rebuild architecture then load weights (Match EfficientNetB3 from notebook)
    try:
        base_model = tf.keras.applications.EfficientNetB3(
            input_shape=(300, 300, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(
                512, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.0005)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(
                256, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.0005)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(6, activation='softmax')
        ])

        model.build((None, 300, 300, 3))
        model.load_weights(model_path, by_name=False)
        print("Model loaded successfully (weight reconstruction).")
        return model
    except Exception as e2:
        print(f"Weight reconstruction also failed: {e2}")
        return None

model = load_waste_model()

# Mapping classes (Alphabetical order from TrashNet/Keras flow_from_directory)
CLASS_LABELS = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

RECYCLING_TIPS = {
    'Cardboard': 'Flatten boxes to save space. Remove any plastic tape or staples.',
    'Glass': 'Rinse containers thoroughly. Labels are usually okay to stay on.',
    'Metal': 'Rinse cans. Crush aluminum cans to save space.',
    'Paper': 'Keep it dry. Do not recycle paper contaminated with food (like pizza boxes).',
    'Plastic': 'Rinse bottles. Check the recycling symbol on the bottom for local compatibility.',
    'Trash': 'This item is non-recyclable. Dispose of it in a standard waste bin.'
}

def predict_label(img_path):
    # Using OpenCV as suggested in the FYP Proposal
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to 300x300 as expected by the EfficientNetB3 model
    img = cv2.resize(img, (300, 300))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Use EfficientNet preprocessing instead of manual / 255.0
    img_array = preprocess_input(img_array)

    # Handle case where model failed to load
    if model is None:
        return "Model Error", 0.0, {}

    predictions = model.predict(img_array)[0]
    class_idx = np.argmax(predictions)
    confidence = float(np.max(predictions))

    all_scores = {CLASS_LABELS[i]: float(predictions[i]) for i in range(len(CLASS_LABELS))}

    # Threshold to handle non-waste or unrecognized objects (FYP Proposal: Edge Cases)
    CONFIDENCE_THRESHOLD = 0.5
    if confidence < CONFIDENCE_THRESHOLD:
        return "Unrecognized", confidence, all_scores

    return CLASS_LABELS[class_idx], confidence, all_scores

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm = request.form['confirm_password']

        # Security: reject any role injection attempt
        if request.form.get('role'):
            flash('Invalid registration attempt.')
            return redirect(url_for('register'))

        if password != confirm:
            flash('Passwords do not match.')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered.')
            return redirect(url_for('register'))

        new_user = User(
            username=username,
            email=email,
            password=password,
            role='user'  # ALWAYS hardcoded, never from form
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. Please log in.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        identifier = request.form['username']
        password = request.form['password']
        user = User.query.filter(
            or_(User.username == identifier, User.email == identifier),
            User.role == 'user'
        ).first()

        if user and user.password == password:
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('user_dashboard'))
        else:
            flash('Invalid credentials.')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('home'))

@app.route('/dashboard')
def user_dashboard():
    if not session.get('user_id'):
        return redirect(url_for('login'))

    logs = Log.query.filter_by(user_id=session['user_id'])\
                    .order_by(Log.timestamp.desc()).all()
    return render_template('user_dashboard.html', logs=logs, username=session['username'])

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

        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload PNG, JPG, or JPEG images.')
            return redirect(request.url)

        if file:
            # Add timestamp to filename to avoid collisions
            original_filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            label, confidence, all_scores = predict_label(filepath)

            # Log to database
            new_log = Log(
                filename=filename,
                prediction=label,
                confidence=confidence,
                user_id=session.get('user_id', None)
            )
            db.session.add(new_log)
            db.session.commit()

            tip = RECYCLING_TIPS.get(label, "No specific recycling tips available for this item.")

            return render_template('classifier.html', filename=filename, label=label,
                                   confidence=f"{confidence*100:.2f}%", tip=tip, all_scores=all_scores)

    return render_template('classifier.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        identifier = request.form['username']
        password = request.form['password']
        user = User.query.filter(
            or_(User.username == identifier, User.email == identifier),
            User.role == 'admin'
        ).first()

        if user and user.password == password:
            session['logged_in'] = True
            session['admin_username'] = user.username
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials.')

    if session.get('logged_in'):
        return redirect(url_for('admin_dashboard'))

    return render_template('admin_login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('admin'))

    # Summary data for dashboard
    total_logs = Log.query.count()
    total_users = User.query.count()

    return render_template('admin_dashboard.html', total_logs=total_logs, total_users=total_users)

@app.route('/admin/logs')
def admin_logs():
    if not session.get('logged_in'):
        return redirect(url_for('admin'))

    logs = Log.query.order_by(Log.timestamp.desc()).all()
    logs_data = []
    for log in logs:
        logs_data.append({
            'id': log.id,
            'filename': log.filename,
            'prediction': log.prediction,
            'confidence': log.confidence,
            'feedback': log.feedback,
            'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    return render_template('admin_logs.html', logs=logs, logs_json=logs_data, classes=CLASS_LABELS)

@app.route('/admin/users')
def admin_users():
    if not session.get('logged_in'):
        return redirect(url_for('admin'))

    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin_users.html', users=users)

@app.route('/admin/add_user', methods=['POST'])
def admin_add_user():
    if not session.get('logged_in'):
        return redirect(url_for('admin'))

    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    role = request.form.get('role', 'user')

    # Only allow valid roles
    if role not in ['admin', 'user']:
        flash('Invalid role specified.')
        return redirect(url_for('admin_users'))

    if User.query.filter_by(username=username).first():
        flash(f'Username {username} already exists.')
        return redirect(url_for('admin_users'))

    if User.query.filter_by(email=email).first():
        flash(f'Email {email} already registered.')
        return redirect(url_for('admin_users'))

    new_user = User(
        username=username,
        email=email,
        password=password,
        role=role
    )
    db.session.add(new_user)
    db.session.commit()
    flash(f'{role.capitalize()} account "{username}" created successfully.')
    return redirect(url_for('admin_users'))

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
def admin_delete_user(user_id):
    if not session.get('logged_in'):
        return redirect(url_for('admin'))

    user = User.query.get_or_404(user_id)

    # Prevent admin from deleting their own account
    if user.username == session.get('admin_username'):
        flash('You cannot delete your own admin account.')
        return redirect(url_for('admin_users'))

    db.session.delete(user)
    db.session.commit()
    flash(f'User "{user.username}" deleted successfully.')
    return redirect(url_for('admin_users'))

@app.route('/admin/export')
def admin_export():
    if not session.get('logged_in'):
        return redirect(url_for('admin'))

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['ID', 'Timestamp', 'Filename', 'Prediction', 'Confidence', 'Feedback', 'User ID'])

    logs = Log.query.all()
    for log in logs:
        cw.writerow([log.id, log.timestamp, log.filename, log.prediction, log.confidence, log.feedback, log.user_id])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=classification_logs.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route('/admin/feedback/<int:log_id>/<string:value>')
def admin_feedback(log_id, value):
    if not session.get('logged_in'):
        return redirect(url_for('admin'))

    log = Log.query.get_or_404(log_id)
    if value in ['Correct', 'Incorrect']:
        log.feedback = value
        db.session.commit()
        flash(f'Feedback updated for log #{log_id}')

    return redirect(url_for('admin_logs'))

@app.route('/admin/upload_dataset', methods=['POST'])
def upload_dataset():
    if not session.get('logged_in'):
        return redirect(url_for('admin'))

    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('admin_logs'))

    file = request.files['file']
    category = request.form.get('category')

    if file.filename == '' or not category:
        flash('No selected file or category')
        return redirect(url_for('admin_logs'))

    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload PNG, JPG, or JPEG images.')
        return redirect(url_for('admin_logs'))

    if file and category in CLASS_LABELS:
        filename = secure_filename(file.filename)
        # Ensure category folder exists
        category_path = os.path.join('static/dataset', category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

        file.save(os.path.join(category_path, filename))
        flash(f'Successfully uploaded {filename} to {category}')

    return redirect(url_for('admin_logs'))

@app.route('/admin/logout')
def admin_logout():
    session.pop('logged_in', None)
    session.pop('admin_username', None)
    return redirect(url_for('admin'))

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
