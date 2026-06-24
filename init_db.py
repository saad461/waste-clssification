import os
import pymysql
from app import app, db, User
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def init_db():
    """
    Initializes the MySQL database, creates tables, and sets up the default admin user.
    """
    print("Initializing Database...")

    DB_USER = os.environ.get('MYSQL_USER', 'root')
    DB_PASSWORD = os.environ.get('MYSQL_PASSWORD', '')
    DB_HOST = os.environ.get('MYSQL_HOST', 'localhost')
    DB_NAME = os.environ.get('MYSQL_DB', 'waste_classification')

    # Step 1: Create MySQL Database if not exists
    try:
        conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD)
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        conn.close()
        print(f"Database '{DB_NAME}' ready.")
    except Exception as e:
        print(f"Error creating database: {e}")
        # If we can't create it, maybe it's because we're using SQLite for testing or something else
        # but the prompt says to update init_db.py for the new schema.

    # Ensure upload folder exists
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
        print("Created static/uploads directory.")

    # Step 2: Initialize tables and default admin
    with app.app_context():
        # Create all tables defined in models
        db.create_all()

        # Check if default admin user exists
        admin_username = 'admin'
        admin_email = 'admin@waste.com'
        admin_password = 'admin123'

        existing_admin = User.query.filter_by(username=admin_username).first()
        if not existing_admin:
            hashed_pw = generate_password_hash(admin_password)
            new_admin = User(
                username=admin_username,
                email=admin_email,
                password=hashed_pw,
                role='admin'
            )
            db.session.add(new_admin)
            db.session.commit()
            print(f"Default admin created successfully.")
            print(f"Username: {admin_username}")
            print(f"Password: {admin_password}")
        else:
            print("Admin user already exists. Skipping creation.")

    print("Database setup complete! You can now run 'python app.py' to start the application.")

if __name__ == '__main__':
    init_db()
