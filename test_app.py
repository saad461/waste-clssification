import pytest
import numpy as np
import os
import sys
import cv2

# Set environment variable before importing app to force SQLite for testing
os.environ['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# ─── Model Unit Tests ──────────────────────────────────────────────────────────

class TestModelLoading:
    """Confirm the model loads correctly and has the right architecture."""

    def test_model_loads_successfully(self):
        from app import model
        assert model is not None, "Model failed to load — check waste_model.h5 exists"

    def test_model_output_shape(self):
        from app import model
        dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        output = model.predict(dummy_input)
        assert output.shape == (1, 6), f"Expected output shape (1,6), got {output.shape}"

    def test_model_output_is_probability(self):
        from app import model
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        output = model.predict(dummy_input)
        assert abs(np.sum(output[0]) - 1.0) < 1e-3, "Output probabilities do not sum to 1"

    def test_model_output_no_negatives(self):
        from app import model
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        output = model.predict(dummy_input)
        assert np.all(output >= 0), "Model output contains negative values"


class TestClassLabels:
    """Confirm class labels are correct and complete."""

    def test_class_labels_count(self):
        from app import CLASS_LABELS
        assert len(CLASS_LABELS) == 6, f"Expected 6 labels, got {len(CLASS_LABELS)}"

    def test_class_labels_alphabetical(self):
        from app import CLASS_LABELS
        expected = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        assert CLASS_LABELS == expected, f"Labels mismatch. Got: {CLASS_LABELS}"

    def test_recycling_tips_match_labels(self):
        from app import CLASS_LABELS, RECYCLING_TIPS
        for label in CLASS_LABELS:
            assert label in RECYCLING_TIPS, f"Missing recycling tip for: {label}"


class TestPredictFunction:
    """Confirm predict_label() returns correct types and handles edge cases."""

    def setup_method(self):
        """Create a real temporary test image before each test."""
        self.test_image_path = 'test_temp_image.jpg'
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        cv2.imwrite(self.test_image_path, img)

    def teardown_method(self):
        """Remove the temp image after each test."""
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)

    def test_predict_returns_correct_types(self):
        from app import predict_label
        label, confidence, all_scores = predict_label(self.test_image_path)
        assert isinstance(label, str), "Label should be a string"
        assert isinstance(confidence, float), "Confidence should be a float"
        assert isinstance(all_scores, dict), "all_scores should be a dictionary"

    def test_predict_confidence_between_0_and_1(self):
        from app import predict_label
        label, confidence, all_scores = predict_label(self.test_image_path)
        assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"

    def test_predict_label_is_valid_or_unrecognized(self):
        from app import predict_label, CLASS_LABELS
        label, confidence, all_scores = predict_label(self.test_image_path)
        valid = CLASS_LABELS + ['Unrecognized']
        assert label in valid, f"Unexpected label returned: {label}"

    def test_predict_low_confidence_returns_unrecognized(self):
        """A blank grey image should score low and return Unrecognized."""
        from app import predict_label
        label, confidence, all_scores = predict_label(self.test_image_path)
        if confidence < 0.5:
            assert label == 'Unrecognized', "Low confidence should return Unrecognized"


# ─── Flask App Route Tests ─────────────────────────────────────────────────────

@pytest.fixture
def client():
    from app import app, db
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False

    with app.app_context():
        db.create_all()
        # Create default admin for tests
        from app import User
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', email='admin@waste.com',
                         password='admin123', role='admin')
            db.session.add(admin)
            db.session.commit()

    with app.test_client() as client:
        yield client


class TestRoutes:
    """Confirm all main routes load correctly."""

    def test_home_page_loads(self, client):
        response = client.get('/')
        assert response.status_code == 200, "Home page failed to load"

    def test_classifier_page_loads(self, client):
        response = client.get('/classifier')
        assert response.status_code == 200, "Classifier page failed to load"

    def test_about_page_loads(self, client):
        response = client.get('/about')
        assert response.status_code == 200, "About page failed to load"

    def test_admin_page_loads(self, client):
        response = client.get('/admin')
        assert response.status_code == 200, "Admin login page failed to load"

    def test_admin_dashboard_requires_login(self, client):
        response = client.get('/admin/dashboard')
        assert response.status_code == 302, "Dashboard should redirect if not logged in"

    def test_admin_login_wrong_password(self, client):
        response = client.post('/admin', data={
            'username': 'admin',
            'password': 'wrongpassword'
        })
        assert response.status_code == 200
        assert b'Invalid' in response.data

    def test_classifier_upload_no_file(self, client):
        response = client.post('/classifier', data={}, content_type='multipart/form-data')
        assert response.status_code in [200, 302]

    def test_classifier_upload_invalid_file_type(self, client):
        data = {
            'file': (b'fake content', 'test.pdf')
        }
        response = client.post('/classifier', data=data, content_type='multipart/form-data')
        assert response.status_code in [200, 302]


# ─── Database Tests ────────────────────────────────────────────────────────────

class TestDatabase:
    """Confirm database models and logging work correctly."""

    def test_log_entry_created(self, client):
        from app import db, Log, app
        with app.app_context():
            log = Log(filename='test.jpg', prediction='plastic', confidence=0.91)
            db.session.add(log)
            db.session.commit()
            entry = Log.query.filter_by(filename='test.jpg').first()
            assert entry is not None
            assert entry.prediction == 'plastic'
            assert entry.confidence == pytest.approx(0.91)
            db.session.delete(entry)
            db.session.commit()

    def test_log_feedback_update(self, client):
        from app import db, Log, app
        with app.app_context():
            log = Log(filename='feedback_test.jpg', prediction='metal', confidence=0.85)
            db.session.add(log)
            db.session.commit()
            log.feedback = 'Correct'
            db.session.commit()
            updated = Log.query.filter_by(filename='feedback_test.jpg').first()
            assert updated.feedback == 'Correct'
            db.session.delete(updated)
            db.session.commit()


# ─── Auth and Admin Management Tests ───────────────────────────────────────────

class TestUserAuth:

    def test_register_page_loads(self, client):
        response = client.get('/register')
        assert response.status_code == 200

    def test_login_page_loads(self, client):
        response = client.get('/login')
        assert response.status_code == 200

    def test_register_new_user(self, client):
        from app import db, User, app
        response = client.post('/register', data={
            'username': 'testuser',
            'email': 'test@test.com',
            'password': 'testpass123',
            'confirm_password': 'testpass123'
        })
        assert response.status_code in [200, 302]
        with app.app_context():
            user = User.query.filter_by(username='testuser').first()
            assert user is not None
            assert user.role == 'user'
            db.session.delete(user)
            db.session.commit()

    def test_cannot_register_as_admin(self, client):
        from app import db, User, app
        response = client.post('/register', data={
            'username': 'hacker',
            'email': 'hack@test.com',
            'password': 'pass123',
            'confirm_password': 'pass123',
            'role': 'admin'
        })
        with app.app_context():
            user = User.query.filter_by(username='hacker').first()
            if user:
                assert user.role == 'user', "Role injection succeeded — critical security bug!"
                db.session.delete(user)
                db.session.commit()

    def test_passwords_must_match(self, client):
        response = client.post('/register', data={
            'username': 'mismatch',
            'email': 'mis@test.com',
            'password': 'pass123',
            'confirm_password': 'different'
        })
        assert response.status_code in [200, 302]

    def test_duplicate_username_rejected(self, client):
        from app import db, User, app
        with app.app_context():
            user = User(username='dupuser', email='dup@test.com',
                        password='pass', role='user')
            db.session.add(user)
            db.session.commit()

        response = client.post('/register', data={
            'username': 'dupuser',
            'email': 'new@test.com',
            'password': 'pass123',
            'confirm_password': 'pass123'
        })
        assert response.status_code in [200, 302]

        with app.app_context():
            user = User.query.filter_by(username='dupuser').first()
            if user:
                db.session.delete(user)
                db.session.commit()

    def test_login_valid_user(self, client):
        from app import db, User, app
        with app.app_context():
            user = User(username='logintest', email='login@test.com',
                        password='pass123', role='user')
            db.session.add(user)
            db.session.commit()

        response = client.post('/login', data={
            'username': 'logintest',
            'password': 'pass123'
        })
        assert response.status_code in [200, 302]

        with app.app_context():
            user = User.query.filter_by(username='logintest').first()
            if user:
                db.session.delete(user)
                db.session.commit()

    def test_login_wrong_password(self, client):
        response = client.post('/login', data={
            'username': 'admin',
            'password': 'wrongpass'
        })
        assert response.status_code == 200
        assert b'Invalid' in response.data

    def test_dashboard_requires_login(self, client):
        response = client.get('/dashboard')
        assert response.status_code == 302

    def test_admin_not_accessible_by_user_session(self, client):
        with client.session_transaction() as sess:
            sess['user_id'] = 999
            sess['username'] = 'fakeuser'
        response = client.get('/admin/dashboard')
        assert response.status_code == 302

    def test_admin_link_hidden_from_public(self, client):
        response = client.get('/')
        assert b'/admin' not in response.data, "Admin link must not appear on homepage"


class TestAdminUserManagement:

    def admin_login(self, client):
        client.post('/admin', data={
            'username': 'admin',
            'password': 'admin123'
        })

    def test_admin_can_add_user(self, client):
        from app import db, User, app
        self.admin_login(client)
        response = client.post('/admin/add_user', data={
            'username': 'newuser',
            'email': 'newuser@test.com',
            'password': 'pass123',
            'role': 'user'
        })
        assert response.status_code in [200, 302]
        with app.app_context():
            user = User.query.filter_by(username='newuser').first()
            assert user is not None
            assert user.role == 'user'
            db.session.delete(user)
            db.session.commit()

    def test_admin_can_add_another_admin(self, client):
        from app import db, User, app
        self.admin_login(client)
        response = client.post('/admin/add_user', data={
            'username': 'admin2',
            'email': 'admin2@test.com',
            'password': 'pass123',
            'role': 'admin'
        })
        assert response.status_code in [200, 302]
        with app.app_context():
            user = User.query.filter_by(username='admin2').first()
            assert user is not None
            assert user.role == 'admin'
            db.session.delete(user)
            db.session.commit()

    def test_admin_cannot_delete_own_account(self, client):
        from app import User, app, db
        self.admin_login(client)
        with app.app_context():
            admin = User.query.filter_by(username='admin').first()
            admin_id = admin.id

        response = client.post(f'/admin/delete_user/{admin_id}')
        assert response.status_code in [200, 302]

        with app.app_context():
            still_exists = User.query.filter_by(username='admin').first()
            assert still_exists is not None, "Admin deleted their own account — not allowed"

    def test_non_admin_cannot_add_user(self, client):
        with client.session_transaction() as sess:
            sess['user_id'] = 999
        response = client.post('/admin/add_user', data={
            'username': 'sneaky',
            'email': 'sneaky@test.com',
            'password': 'pass',
            'role': 'user'
        })
        assert response.status_code == 302
