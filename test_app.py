import unittest
from unittest.mock import MagicMock
import sys

# Mock tensorflow and other heavy imports before importing app
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.keras'] = MagicMock()
sys.modules['tensorflow.keras.applications'] = MagicMock()
sys.modules['tensorflow.keras.models'] = MagicMock()
sys.modules['tensorflow.keras.layers'] = MagicMock()

from app import app, db, User, Log

class WasteAppTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app = app.test_client()
        with app.app_context():
            db.create_all()

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.create_all()

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Automated Waste Classification System', response.data)

    def test_classifier_page_get(self):
        response = self.app.get('/classifier')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Upload Waste Image', response.data)

    def test_about_page(self):
        response = self.app.get('/about')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'About the Project', response.data)

    def test_admin_login_page(self):
        response = self.app.get('/admin')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Admin Login', response.data)

    def test_admin_dashboard_access_denied(self):
        response = self.app.get('/admin/dashboard', follow_redirects=True)
        self.assertIn(b'Admin Login', response.data)

    def test_about_page_visualizations(self):
        response = self.app.get('/about')
        self.assertIn(b'training_metrics.png', response.data)
        self.assertIn(b'confusion_matrix.png', response.data)

if __name__ == '__main__':
    unittest.main()
