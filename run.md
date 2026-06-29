# A to Z Guide: Running Waste Classification System on Your Laptop

This guide provides step-by-step instructions to set up and run the Waste Classification project from scratch.

---

## 1. Prerequisites
Before starting, ensure you have the following installed on your laptop:
- **Python 3.8 or higher:** Download from [python.org](https://www.python.org/downloads/)
- **XAMPP (for MySQL):** Download from [apachefriends.org](https://www.apachefriends.org/download.html)
- **Web Browser:** Chrome, Edge, or Firefox.

---

## 2. Step-by-Step Setup

### Step A: Prepare the Project Folder
1. Extract the project ZIP file (if you have one) or clone the repository to a folder on your laptop.
2. Open your terminal (Command Prompt or PowerShell on Windows; Terminal on Mac/Linux) and navigate to this folder.

### Step B: Database Setup (XAMPP & MySQL)
1. Open the **XAMPP Control Panel**.
2. Click **Start** next to the **MySQL** module. (You can also start Apache if you want to use phpMyAdmin).
3. Open your browser and go to: `http://localhost/phpmyadmin`
4. Click on the **Import** tab at the top.
5. Click **Choose File** and select the `database.sql` file located in the root of the project folder.
6. Scroll down and click **Import** (or **Go**).
7. The database `waste_classification` and required tables will be created automatically.

### Step C: Create a Virtual Environment
It is recommended to run the project in a virtual environment to keep dependencies organized.
1. In your terminal, run:
   ```bash
   python -m venv venv
   ```
2. Activate the virtual environment:
   - **Windows:** `venv\Scripts\activate`
   - **Mac/Linux:** `source venv/bin/activate`

### Step D: Install Dependencies
Install all required libraries using the provided requirements file:
```bash
pip install -r requirements.txt
```

### Step E: Environment Configuration
1. Look for a file named `.env.example` in the project root.
2. Rename or copy it to a new file named `.env`.
3. Open `.env` in a text editor and ensure the database settings match your XAMPP defaults:
   - `MYSQL_USER=root`
   - `MYSQL_PASSWORD=` (leave empty for default XAMPP)
   - `MYSQL_HOST=localhost`
   - `MYSQL_DB=waste_classification`

---

## 3. Running the Application

### Start the Flask Server
In your terminal (with the virtual environment activated), run:
```bash
python app.py
```

### Access the Website
1. Open your browser and go to: `http://127.0.0.1:5000`
2. You can now use the **Classifier** to upload images of waste for identification.

---

## 4. Admin and User Access

### For Regular Users:
- Go to `/register` to create an account.
- Go to `/login` to sign in and view your personal classification history on the **Dashboard**.

### For Admins (Hidden Panel):
- URL: `http://127.0.0.1:5000/admin`
- **Default Admin Credentials:**
  - **Username:** `admin`
  - **Password:** `admin123`
- Inside the Admin Panel, you can manage users, view all system logs, export data to CSV, and upload new images to the training dataset.

---

## 5. Troubleshooting
- **Database Connection Error:** Ensure MySQL is started in XAMPP and your `.env` credentials are correct.
- **Missing Module Error:** Run `pip install -r requirements.txt` again to ensure all libraries are installed.
- **Model Loading Error:** Ensure `waste_model.h5` is present in the root directory.

---
**Happy Recycling!** 🌍
