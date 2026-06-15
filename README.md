# Waste Classification (FYP Project)

This project is an automated Waste Classification System developed for a Final Year Project (FYP). It uses a Convolutional Neural Network (CNN) based on the **MobileNetV2** architecture to classify waste into six categories: Cardboard, Glass, Metal, Paper, Plastic, and Organic Material.

## Features
- **Real-time Classification:** Upload an image and get instant prediction with a confidence score.
- **Admin Dashboard:** Monitor system usage logs and manage the dataset categories.
- **Dataset Management:** Admins can upload new images to specific categories to improve the model.
- **Evaluation Visualizations:** Detailed training metrics and confusion matrix displayed on the About page for academic evaluation.
- **Secure & Robust:** Protected by CSRF security and robust model loading across different Keras versions.

---

## 🛠 Local Setup Instructions

Follow these steps to run the project on your local computer:

### 1. Prerequisites
Ensure you have **Python 3.8 or higher** installed on your system. You can check your version by running:
```bash
python --version
```

### 2. Clone or Extract the Project
Extract the project files into a folder of your choice. Open your terminal or command prompt in that folder.

### 3. Create a Virtual Environment (Recommended)
It is best practice to use a virtual environment to keep your dependencies isolated.
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
Install all necessary Python libraries using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 5. Initialize the Database
Before running the app for the first time, you should initialize the database to create the tables and the default admin account:
```bash
python init_db.py
```
This will create a `waste_management.db` file in your project directory.

### 6. Run the Application
Start the Flask development server:
```bash
python app.py
```
The application will be available at: **`http://127.0.0.1:5000`**

---

## 🔐 Admin Access
To access the admin panel for monitoring and dataset management:
- **URL:** `http://127.0.0.1:5000/admin`
- **Username:** `admin`
- **Password:** `admin123` (You can change this in `init_db.py` before setup)

---

## 🛡️ Security Configuration
For a production or final submission, you should change the `SECRET_KEY` in `app.py`.
- Open `app.py` and find the line: `app.secret_key = os.environ.get('SECRET_KEY', '...')`.
- Change the fallback string to any long random sequence of characters.

---

## 📂 Project Structure
- `app.py`: The main Flask backend handles routing, database, and model inference.
- `init_db.py`: Database setup script to create tables and admin user.
- `waste_management.db`: The SQLite database file (appears after running `init_db.py`).
- `static/dataset/`: **This is where new images uploaded via the Admin Panel are stored.** They are organized into folders by category (e.g., `static/dataset/Plastic/`).
- `waste_model.h5`: The trained weights for the CNN model.
- `templates/`: HTML files for the web interface.
- `static/`: Contains CSS, images, and uploaded files.
- `static/dataset/`: Organized folder structure for the training dataset.
- `requirements.txt`: List of Python libraries required.
- `test_app.py`: Unit tests for the application.

---

## 🔄 Retraining the Model
If you upload many new images via the Admin Panel and want to improve the model:
1. Collect the new images from the `static/dataset/` folder.
2. Add them to your training dataset in the `waste.ipynb` notebook.
3. Run the training cells in the notebook to generate a new `waste_model.h5`.
4. Replace the old `waste_model.h5` in the root directory with your new one.

---

## 📈 Model Performance
- **Training Accuracy:** 96.7%
- **Validation Accuracy:** 76.7%
- **Architecture:** MobileNetV2 (Transfer Learning)

---
**Developed by:** [Your Name]
**Supervisor:** Iqra Iqbal Khan
