CREATE DATABASE IF NOT EXISTS waste_classification;
USE waste_classification;

CREATE TABLE IF NOT EXISTS user (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(200) NOT NULL,
    role VARCHAR(10) DEFAULT 'user' NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(100) NOT NULL,
    prediction VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    feedback VARCHAR(20),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_id INT,
    FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE SET NULL
);

-- Default admin account
INSERT INTO user (username, email, password, role)
VALUES ('admin', 'admin@waste.com', 'admin123', 'admin');
