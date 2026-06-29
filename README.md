# Waste Classification (FYP Project)

This project is an automated Waste Classification System developed for a Final Year Project (FYP). It uses a Convolutional Neural Network (CNN) based on the **MobileNetV2** architecture to classify waste into six categories: cardboard, glass, metal, paper, plastic, and trash.

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
Ensure you have **Python 3.8 or higher** installed on your system.
You also need **XAMPP** installed for the MySQL database.

### 2. Database Setup:
1. Open XAMPP Control Panel and start MySQL
2. Go to http://localhost/phpmyadmin
3. Click Import tab
4. Select database.sql from the project root
5. Click Go
6. Database is ready — run python app.py

### 3. Clone or Extract the Project
Extract the project files into a folder of your choice. Open your terminal or command prompt in that folder.

### 4. Create a Virtual Environment (Recommended)
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 5. Install Dependencies
```bash
pip install -r requirements.txt
```

### 6. Environment Configuration
1. Copy `.env.example` to a new file named `.env`.
2. Update the values in `.env` if your MySQL configuration differs from XAMPP defaults.
   - `MYSQL_USER=root`
   - `MYSQL_PASSWORD=` (leave blank for XAMPP)
   - `MYSQL_HOST=localhost`
   - `MYSQL_DB=waste_classification`

### 7. Run the Application
Start the Flask development server:
```bash
python app.py
```
The application will be available at: **`http://127.0.0.1:5000`**

---

## 📂 Project Structure
- `app.py`: The main Flask backend handles routing, database, and model inference.
- `database.sql`: SQL script to manually set up the MySQL database and tables.
- `.env`: Environment variables for database and security (You must create this).
- `static/dataset/`: Organized folder structure for the training dataset.
- `waste_model.h5`: The trained weights for the CNN model.
- `templates/`: HTML files for the web interface.
- `static/`: Contains CSS, images, and uploaded files.
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
**Developed by:** Saad Nadeem
**Supervisor:** Iqra Iqbal Khan
