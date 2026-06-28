from app import app, db, User

def init_db():
    """Initializes the database with default users for screenshot generation."""
    with app.app_context():
        # Create tables
        db.create_all()

        # Add default admin if not exists
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin',
                email='admin@example.com',
                password='admin123',
                role='admin'
            )
            db.session.add(admin)

        # Add default user if not exists
        if not User.query.filter_by(username='testuser').first():
            user = User(
                username='testuser',
                email='user@example.com',
                password='password123',
                role='user'
            )
            db.session.add(user)

        db.session.commit()
        print("Database initialized with default users.")

if __name__ == "__main__":
    init_db()
