import os
from app import app, db, User
from werkzeug.security import generate_password_hash

def init_db():
    """
    Initializes the SQLite database, creates tables, and sets up the default admin user.
    """
    print("Initializing Database...")

    # Ensure upload folder exists
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
        print("Created static/uploads directory.")

    with app.app_context():
        # Create all tables defined in models
        db.create_all()

        # Check if default admin user exists
        admin_username = 'admin'
        admin_password = 'admin123'

        existing_admin = User.query.filter_by(username=admin_username).first()
        if not existing_admin:
            hashed_pw = generate_password_hash(admin_password)
            new_admin = User(username=admin_username, password=hashed_pw)
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
