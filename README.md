# EduPortal — Student Performance & School Management System 🎓

An advanced, ML-powered School Management and Student Performance Analytics Web Application built with **Flask**, **scikit-learn**, and **MongoDB**. Features an executive dark-themed user interface, single-student performance evaluation, multi-student bulk batch evaluation via CSV & Excel, and automated reporting.

---

## ✨ Features

- **Executive Dark UI**: Sleek, modern interface with consistent layout, typography, navigation, and interactive charts across all pages.
- **Machine Learning Pass/Fail Prediction**: Evaluates 4 key academic indicators:
  - Study Hours per day
  - Attendance %
  - Previous Exam Marks
  - Assignment Marks
- **Multi-Student Batch Prediction**: Upload `.csv` or `.xlsx` files with dozens/hundreds of students to predict outcomes simultaneously with confidence scores.
- **Export Reports**: Download prediction results and archives in both **Excel (.xlsx)** and **CSV** formats.
- **Role-Based Access Control**:
  - **Admin**: Student directory management, account enable/disable, school-wide performance metrics, all prediction records.
  - **Student**: Personal performance dashboard, prediction tool, prediction history, profile management.
- **Database Integration**: Cloud MongoDB integration for persistent student and prediction storage.

---

## 🛠️ Technology Stack

- **Backend**: Python 3, Flask
- **Machine Learning**: scikit-learn, NumPy, Pandas
- **Database**: MongoDB (PyMongo)
- **Spreadsheet Processing**: openpyxl, pandas
- **Frontend**: HTML5, Vanilla CSS3 (Custom Design System), JavaScript

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Unzela-Waseem/SchoolManagementSystemWebapp.git
cd SchoolManagementSystemWebapp
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your MongoDB connection details:
```bash
cp .env.example .env
```

### 5. Run the Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 📂 Project Structure
```
├── app.py                         # Main Flask application & routes
├── student_pass_fail_model.pkl    # Pre-trained ML model pipeline
├── students_sample_data.csv       # Sample batch evaluation data (CSV)
├── students_sample_data.xlsx      # Sample batch evaluation data (Excel)
├── requirements.txt               # Project dependencies
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git ignore rules (protects .env & venv)
├── static/
│   ├── style.css                  # Executive EduPortal dark design system
│   └── main.js                    # Frontend interactions
└── templates/
    ├── base.html                  # Master layout with sidebar & topbar
    ├── login.html                 # Sign-in authentication
    ├── signup.html                # Student registration
    ├── student_dashboard.html     # Student analytics overview
    ├── admin_dashboard.html       # School management & metrics overview
    ├── index.html                 # Single prediction tool
    ├── batch_prediction.html      # Batch CSV/Excel prediction tool
    ├── student_history.html       # Student prediction archive
    ├── admin_users.html           # Student accounts management
    ├── admin_history.html         # All prediction records
    └── error.html                 # Error handler page
```

---

## 🔒 Security Notice
Actual database credentials, passwords, and `.env` files are kept confidential and are excluded via `.gitignore`.
