import unittest
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

if __name__ == '__main__':
    unittest.main()
