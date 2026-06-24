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
        expected = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
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

    def test_predict_returns_string_and_float(self):
        from app import predict_label
        label, confidence = predict_label(self.test_image_path)
        assert isinstance(label, str), "Label should be a string"
        assert isinstance(confidence, float), "Confidence should be a float"

    def test_predict_confidence_between_0_and_1(self):
        from app import predict_label
        label, confidence = predict_label(self.test_image_path)
        assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"

    def test_predict_label_is_valid_or_unrecognized(self):
        from app import predict_label, CLASS_LABELS
        label, confidence = predict_label(self.test_image_path)
        valid = CLASS_LABELS + ['Unrecognized']
        assert label in valid, f"Unexpected label returned: {label}"

    def test_predict_low_confidence_returns_unrecognized(self):
        """A blank grey image should score low and return Unrecognized."""
        from app import predict_label
        label, confidence = predict_label(self.test_image_path)
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
            log = Log(filename='test.jpg', prediction='Plastic', confidence=0.91)
            db.session.add(log)
            db.session.commit()
            entry = Log.query.filter_by(filename='test.jpg').first()
            assert entry is not None
            assert entry.prediction == 'Plastic'
            assert entry.confidence == pytest.approx(0.91)
            db.session.delete(entry)
            db.session.commit()

    def test_log_feedback_update(self, client):
        from app import db, Log, app
        with app.app_context():
            log = Log(filename='feedback_test.jpg', prediction='Metal', confidence=0.85)
            db.session.add(log)
            db.session.commit()
            log.feedback = 'Correct'
            db.session.commit()
            updated = Log.query.filter_by(filename='feedback_test.jpg').first()
            assert updated.feedback == 'Correct'
            db.session.delete(updated)
            db.session.commit()
