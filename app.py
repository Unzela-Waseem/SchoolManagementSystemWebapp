
import io
import os
import pickle
from datetime import datetime, timezone
from functools import wraps

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, abort, send_file, make_response
)
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, PyMongoError
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
from bson.errors import InvalidId

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-secret-key")

# ---------------- MongoDB CONNECTION ----------------

MONGO_URI = os.getenv("MONGO_URI", "").strip()
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "student_performance").strip()

client = None
db = None
users = None
predictions = None

try:
    if not MONGO_URI:
        raise ValueError("MONGO_URI missing from .env")

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")

    db = client[MONGO_DB_NAME]
    users = db["users"]
    predictions = db["predictions"]

    users.create_index("email", unique=True)
    predictions.create_index([
        ("user_id", ASCENDING),
        ("created_at", DESCENDING)
    ])

    print("MongoDB connected:", MONGO_DB_NAME)

except Exception as exc:
    app.logger.exception("MongoDB connection failed: %s", exc)


def database_ready():
    return users is not None and predictions is not None


# ---------------- LOAD MACHINE LEARNING MODEL ----------------

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "student_pass_fail_model.pkl"
)

model = None

feature_columns = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Assignment_Marks"
]

class_labels = {}

try:
    with open(MODEL_PATH, "rb") as file:
        saved_model = pickle.load(file)

    if isinstance(saved_model, dict):
        model = saved_model.get("model")
        feature_columns = saved_model.get(
            "feature_columns",
            feature_columns
        )
        class_labels = saved_model.get("class_labels", {})
    else:
        model = saved_model

    if model is None:
        raise ValueError("Model was not found inside pickle file.")

    feature_columns = list(feature_columns)

    # Use the feature names stored by the trained model if available.
    if hasattr(model, "feature_names_in_"):
        saved_features = list(model.feature_names_in_)
        if len(saved_features) == len(feature_columns):
            feature_columns = saved_features

    print("ML model loaded successfully.")
    print("Model file:", MODEL_PATH)
    print("Input features:", feature_columns)
    print("Model classes:", getattr(model, "classes_", "Unavailable"))

except Exception as exc:
    app.logger.exception("Model loading failed: %s", exc)


# ---------------- PREDICTION LABEL HELPER ----------------

def readable_prediction(raw_prediction):
    """
    Converts common numeric or text labels into Pass/Fail.
    The 0/1 mapping assumes the training labels were:
    0 = Fail and 1 = Pass.
    """
    key = str(raw_prediction).strip()
    mapped = class_labels.get(key, raw_prediction)

    if isinstance(mapped, (int, np.integer)):
        if int(mapped) == 1:
            return "Pass"
        if int(mapped) == 0:
            return "Fail"

    if isinstance(mapped, (float, np.floating)) and float(mapped).is_integer():
        if int(mapped) == 1:
            return "Pass"
        if int(mapped) == 0:
            return "Fail"

    label = str(mapped).strip().lower()

    if label in {"pass", "passed", "true", "yes"}:
        return "Pass"

    if label in {"fail", "failed", "false", "no"}:
        return "Fail"

    # Keep unknown labels visible instead of inventing an outcome.
    return str(mapped)


# ---------------- INITIAL ADMIN ----------------

def create_initial_admin():
    if not database_ready():
        return

    email = os.getenv("ADMIN_EMAIL", "").strip().lower()
    password = os.getenv("ADMIN_PASSWORD", "")
    name = os.getenv("ADMIN_NAME", "System Admin").strip()

    if not email or not password:
        print("Admin seed skipped: configure ADMIN_EMAIL and ADMIN_PASSWORD.")
        return

    existing = users.find_one({"email": email})

    if existing:
        if existing.get("role") != "admin":
            print("WARNING: ADMIN_EMAIL belongs to a non-admin account.")
        return

    users.insert_one({
        "name": name or "System Admin",
        "email": email,
        "password": generate_password_hash(password),
        "role": "admin",
        "active": True,
        "created_at": datetime.now(timezone.utc)
    })

    print("Initial Admin account created.")


if database_ready():
    try:
        create_initial_admin()
    except Exception:
        app.logger.exception("Admin initialization failed.")


# ---------------- AUTHENTICATION HELPERS ----------------

def current_user():
    if not database_ready():
        return None

    user_id = session.get("user_id")

    if not user_id:
        return None

    try:
        user = users.find_one({"_id": ObjectId(user_id)})
    except (InvalidId, TypeError):
        session.clear()
        return None
    except PyMongoError:
        app.logger.exception("Could not retrieve user.")
        return None

    if not user or not user.get("active", True):
        session.clear()
        return None

    return user


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not database_ready():
            flash("Database unavailable. Check your .env settings.", "danger")
            return redirect(url_for("login"))

        if not current_user():
            session.clear()
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))

        return view(*args, **kwargs)

    return wrapped


def role_required(required_role):
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            user = current_user()

            if not user:
                session.clear()
                flash("Please log in first.", "warning")
                return redirect(url_for("login"))

            if user.get("role") != required_role:
                flash("You are not authorized to access this page.", "danger")
                return redirect(url_for("dashboard"))

            return view(*args, **kwargs)

        return wrapped

    return decorator


@app.context_processor
def inject_user():
    return {"current_user": current_user()}


# ---------------- HOME AND DASHBOARD ----------------

@app.route("/")
def home():
    if current_user():
        return redirect(url_for("dashboard"))

    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()

    if user.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))

    return redirect(url_for("student_dashboard"))


# ---------------- LOGIN ----------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        if not database_ready():
            flash("Database connection failed.", "danger")
            return render_template("login.html")

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        try:
            user = users.find_one({"email": email}) if email else None
        except PyMongoError:
            app.logger.exception("Login query failed.")
            flash("Could not access database.", "danger")
            return render_template("login.html")

        if not user or not check_password_hash(
            user.get("password", ""), password
        ):
            flash("Invalid email or password.", "danger")
            return render_template("login.html")

        if not user.get("active", True):
            flash("Your account is disabled.", "danger")
            return render_template("login.html")

        session.clear()
        session["user_id"] = str(user["_id"])

        flash("Login successful.", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html")


# ---------------- STUDENT SIGNUP ----------------

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not database_ready():
            flash("Database connection failed.", "danger")
        elif len(name) < 2:
            flash("Enter your full name.", "danger")
        elif "@" not in email or "." not in email.split("@")[-1]:
            flash("Enter a valid email address.", "danger")
        elif len(password) < 8:
            flash("Password must contain at least 8 characters.", "danger")
        elif password != confirm_password:
            flash("Passwords do not match.", "danger")
        else:
            try:
                users.insert_one({
                    "name": name,
                    "email": email,
                    "password": generate_password_hash(password),
                    "role": "student",
                    "active": True,
                    "created_at": datetime.now(timezone.utc)
                })

                flash("Account created. Please log in.", "success")
                return redirect(url_for("login"))

            except DuplicateKeyError:
                flash("Email already registered.", "danger")
            except PyMongoError:
                app.logger.exception("Signup database error.")
                flash("Could not create account.", "danger")

    return render_template("signup.html")


# ---------------- LOGOUT ----------------

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))


# ---------------- STUDENT DASHBOARD ----------------

@app.route("/student/dashboard")
@login_required
@role_required("student")
def student_dashboard():
    user = current_user()
    query = {"user_id": str(user["_id"])}

    total = predictions.count_documents(query)
    recent = list(
        predictions.find(query)
        .sort("created_at", DESCENDING)
        .limit(5)
    )

    return render_template(
        "student_dashboard.html",
        total=total,
        recent=recent
    )


# ---------------- STUDENT PREDICTION ----------------

@app.route("/student/index", methods=["GET", "POST"])
@login_required
@role_required("student")
def student_index():
    user = current_user()
    input_values = {}
    result = None

    if request.method == "POST":
        input_values = {
            feature: request.form.get(feature, "").strip()
            for feature in feature_columns
        }

        if model is None:
            flash("ML model failed to load. Check the terminal.", "danger")
        else:
            try:
                values = []

                for feature in feature_columns:
                    raw_value = input_values.get(feature, "")

                    if not raw_value:
                        raise ValueError(
                            f"{feature.replace('_', ' ')} is required."
                        )

                    value = float(raw_value)

                    if not np.isfinite(value):
                        raise ValueError("Enter valid numeric values.")

                    values.append(value)

                values_by_name = dict(zip(feature_columns, values))

                if "Study_Hours" in values_by_name:
                    hours = values_by_name["Study_Hours"]
                    if not 0 <= hours <= 24:
                        raise ValueError(
                            "Study Hours must be between 0 and 24."
                        )

                for feature in (
                    "Attendance",
                    "Previous_Marks",
                    "Assignment_Marks"
                ):
                    if feature in values_by_name:
                        score = values_by_name[feature]
                        if not 0 <= score <= 100:
                            raise ValueError(
                                f"{feature.replace('_', ' ')} "
                                "must be between 0 and 100."
                            )

                sample = np.asarray([values], dtype=float)

                # Actual ML prediction
                raw_prediction = model.predict(sample)[0]
                prediction_text = readable_prediction(raw_prediction)

                # Probability of each class, if supported
                confidence = None
                probabilities_by_class = {}

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(sample)[0]
                    classes = getattr(model, "classes_", [])

                    for class_value, probability in zip(
                        classes, probabilities
                    ):
                        class_name = readable_prediction(class_value)
                        probabilities_by_class[class_name] = round(
                            float(probability) * 100, 2
                        )

                    # Confidence for the predicted class
                    raw_key = str(raw_prediction).strip()
                    try:
                        predicted_index = list(classes).index(raw_prediction)
                        confidence = round(
                            float(probabilities[predicted_index]) * 100,
                            2
                        )
                    except (ValueError, IndexError):
                        confidence = round(
                            float(np.max(probabilities)) * 100, 2
                        )

                record = {
                    "user_id": str(user["_id"]),
                    "student_name": user["name"],
                    "student_email": user["email"],
                    "inputs": values_by_name,
                    "prediction": prediction_text,
                    "raw_prediction": str(raw_prediction),
                    "confidence": confidence,
                    "class_probabilities": probabilities_by_class,
                    "created_at": datetime.now(timezone.utc)
                }

                # Save prediction in MongoDB
                saved = predictions.insert_one(record)
                record["_id"] = saved.inserted_id
                result = record

                flash(
                    f"Prediction complete: {prediction_text.upper()}",
                    "success"
                )

            except (ValueError, TypeError) as exc:
                flash(str(exc), "danger")
            except PyMongoError:
                app.logger.exception("Could not save prediction.")
                flash(
                    "Prediction could not be saved to MongoDB.",
                    "danger"
                )
            except Exception:
                app.logger.exception("Prediction failed.")
                flash(
                    "Prediction failed. Check the terminal for details.",
                    "danger"
                )

    return render_template(
        "index.html",
        features=feature_columns,
        input_values=input_values,
        result=result
    )


# ---------------- STUDENT HISTORY ----------------

@app.route("/student/history")
@login_required
@role_required("student")
def student_history():
    user = current_user()

    records = list(
        predictions.find({"user_id": str(user["_id"])})
        .sort("created_at", DESCENDING)
        .limit(100)
    )

    return render_template("student_history.html", records=records)


# ---------------- DOWNLOAD PREDICTION (CSV / EXCEL) ----------------

def _build_prediction_df(user):
    """Return a DataFrame of all predictions for the given user."""
    records = list(
        predictions.find({"user_id": str(user["_id"])})
        .sort("created_at", DESCENDING)
        .limit(1000)
    )

    rows = []
    for rec in records:
        inputs = rec.get("inputs", {})
        row = {
            "Student Name":   rec.get("student_name", ""),
            "Student Email":  rec.get("student_email", ""),
            "Study Hours":    inputs.get("Study_Hours", ""),
            "Attendance":     inputs.get("Attendance", ""),
            "Previous Marks": inputs.get("Previous_Marks", ""),
            "Assignment Marks": inputs.get("Assignment_Marks", ""),
            "Prediction":     rec.get("prediction", ""),
            "Confidence (%)": rec.get("confidence", ""),
            "Date":           rec.get("created_at", ""),
        }
        rows.append(row)

    return pd.DataFrame(rows)


@app.route("/student/prediction/download/csv")
@login_required
@role_required("student")
def download_prediction_csv():
    user = current_user()
    df = _build_prediction_df(user)

    if df.empty:
        flash("Koi prediction record nahi mila download ke liye.", "warning")
        return redirect(url_for("student_history"))

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    filename = f"predictions_{user['name'].replace(' ', '_')}.csv"

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    response.headers["Content-Type"] = "text/csv"
    return response


@app.route("/student/prediction/download/excel")
@login_required
@role_required("student")
def download_prediction_excel():
    user = current_user()
    df = _build_prediction_df(user)

    if df.empty:
        flash("Koi prediction record nahi mila download ke liye.", "warning")
        return redirect(url_for("student_history"))

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    output.seek(0)

    filename = f"predictions_{user['name'].replace(' ', '_')}.xlsx"

    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=filename
    )


# ---------------- BATCH PREDICTION (CSV / EXCEL UPLOAD) ----------------

batch_cache = {}


def normalize_column_name(col):
    c = str(col).strip().lower().replace(" ", "_").replace("-", "_")
    if c in {"study_hours", "studyhours", "study_hour", "hours_studied", "study"}:
        return "Study_Hours"
    if c in {"attendance", "attendance_percentage", "attendance_percent", "att"}:
        return "Attendance"
    if c in {"previous_marks", "previousmarks", "prev_marks", "prevmarks", "previous_score"}:
        return "Previous_Marks"
    if c in {"assignment_marks", "assignmentmarks", "assignment_score", "assign_marks"}:
        return "Assignment_Marks"
    if c in {"name", "student_name", "student", "full_name", "candidate_name"}:
        return "Student_Name"
    if c in {"roll_no", "rollno", "roll_number", "id", "student_id", "reg_no"}:
        return "Roll_No"
    if c in {"email", "student_email", "mail"}:
        return "Student_Email"
    return col


@app.route("/batch-prediction", methods=["GET", "POST"])
@login_required
def batch_prediction():
    user = current_user()
    summary = None
    results = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            flash("Kripya CSV ya Excel (.xlsx, .xls) file upload karein.", "danger")
            return redirect(url_for("batch_prediction"))

        filename = file.filename.lower()
        if not (filename.endswith(".csv") or filename.endswith(".xlsx") or filename.endswith(".xls")):
            flash("Sirf CSV ya Excel (.xlsx / .xls) file upload ki ja sakti hai.", "danger")
            return redirect(url_for("batch_prediction"))

        if model is None:
            flash("ML model load nahi hai. Server logs check karein.", "danger")
            return redirect(url_for("batch_prediction"))

        try:
            if filename.endswith(".csv"):
                try:
                    df = pd.read_csv(file)
                except UnicodeDecodeError:
                    file.seek(0)
                    df = pd.read_csv(file, encoding="latin1")
            else:
                df = pd.read_excel(file)

            if df.empty:
                flash("Uploaded file bilkul khali (empty) hai.", "warning")
                return redirect(url_for("batch_prediction"))

            # Normalize column names
            df.columns = [normalize_column_name(c) for c in df.columns]

            required_features = ["Study_Hours", "Attendance", "Previous_Marks", "Assignment_Marks"]
            missing_cols = [c for c in required_features if c not in df.columns]

            if missing_cols:
                missing_str = ", ".join(missing_cols)
                flash(
                    f"File mein required columns missing hain: {missing_str}. "
                    "Neeche diye gaye Sample Template se check karein.",
                    "danger"
                )
                return redirect(url_for("batch_prediction"))

            # Convert to numeric
            for col in required_features:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            valid_rows = df[required_features].notna().all(axis=1)
            if not valid_rows.any():
                flash("File mein koi bhi valid numeric student record nahi mila.", "danger")
                return redirect(url_for("batch_prediction"))

            # Optional student columns
            if "Student_Name" not in df.columns:
                df["Student_Name"] = [f"Student {i+1}" for i in range(len(df))]
            else:
                df["Student_Name"] = df["Student_Name"].fillna("Student").astype(str)

            if "Roll_No" not in df.columns:
                df["Roll_No"] = [f"ST-{100 + i + 1}" for i in range(len(df))]
            else:
                df["Roll_No"] = df["Roll_No"].fillna("—").astype(str)

            # Prediction with ML model
            X = df.loc[valid_rows, required_features].copy()
            raw_preds = model.predict(X)
            pred_labels = [readable_prediction(p) for p in raw_preds]

            confidences = []
            if hasattr(model, "predict_proba"):
                classes = list(getattr(model, "classes_", []))
                probas = model.predict_proba(X)
                for rp, prob in zip(raw_preds, probas):
                    try:
                        idx = classes.index(rp)
                        conf = round(float(prob[idx]) * 100, 2)
                    except (ValueError, IndexError):
                        conf = round(float(np.max(prob)) * 100, 2)
                    confidences.append(conf)
            else:
                confidences = [None] * len(pred_labels)

            df.loc[valid_rows, "Prediction"] = pred_labels
            df.loc[valid_rows, "Confidence"] = confidences

            df.loc[~valid_rows, "Prediction"] = "Error (Invalid Data)"
            df.loc[~valid_rows, "Confidence"] = None

            total_evaluated = int(valid_rows.sum())
            pass_count = int((df.loc[valid_rows, "Prediction"] == "Pass").sum())
            fail_count = int((df.loc[valid_rows, "Prediction"] == "Fail").sum())
            pass_rate = round((pass_count / total_evaluated * 100), 1) if total_evaluated > 0 else 0

            valid_conf = [c for c in confidences if c is not None]
            avg_conf = round(float(np.mean(valid_conf)), 1) if valid_conf else 0

            summary = {
                "total": total_evaluated,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "pass_rate": pass_rate,
                "avg_confidence": avg_conf,
                "filename": file.filename
            }

            # Persist to MongoDB
            if database_ready():
                batch_records = []
                now = datetime.now(timezone.utc)
                for _, row in df[valid_rows].iterrows():
                    batch_records.append({
                        "user_id": str(user["_id"]),
                        "student_name": str(row["Student_Name"]),
                        "student_email": user.get("email", ""),
                        "roll_no": str(row["Roll_No"]),
                        "inputs": {
                            "Study_Hours": float(row["Study_Hours"]),
                            "Attendance": float(row["Attendance"]),
                            "Previous_Marks": float(row["Previous_Marks"]),
                            "Assignment_Marks": float(row["Assignment_Marks"]),
                        },
                        "prediction": str(row["Prediction"]),
                        "confidence": float(row["Confidence"]) if row["Confidence"] is not None else None,
                        "is_batch": True,
                        "created_at": now
                    })
                if batch_records:
                    predictions.insert_many(batch_records)

            batch_cache[str(user["_id"])] = df
            results = df.to_dict(orient="records")

            flash(f"Batch prediction complete: {total_evaluated} students process ho gaye! (Pass: {pass_count}, Fail: {fail_count})", "success")

        except Exception as exc:
            app.logger.exception("Batch prediction error: %s", exc)
            flash(f"File process karne mein masla aaya: {str(exc)}", "danger")

    return render_template(
        "batch_prediction.html",
        summary=summary,
        results=results
    )


@app.route("/batch-prediction/download/csv")
@login_required
def download_batch_csv():
    user = current_user()
    df = batch_cache.get(str(user["_id"]))
    if df is None or df.empty:
        flash("Koi batch prediction data download ke liye maujood nahi hai.", "warning")
        return redirect(url_for("batch_prediction"))

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    filename = f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    response.headers["Content-Type"] = "text/csv"
    return response


@app.route("/batch-prediction/download/excel")
@login_required
def download_batch_excel():
    user = current_user()
    df = batch_cache.get(str(user["_id"]))
    if df is None or df.empty:
        flash("Koi batch prediction data download ke liye maujood nahi hai.", "warning")
        return redirect(url_for("batch_prediction"))

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Batch Predictions")
    output.seek(0)

    filename = f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=filename
    )


@app.route("/batch-prediction/sample/csv")
@login_required
def download_sample_csv():
    sample_df = pd.DataFrame([
        {"Student_Name": "Zainab Fatima",  "Roll_No": "ST-101", "Study_Hours": 6.5, "Attendance": 88, "Previous_Marks": 82, "Assignment_Marks": 85},
        {"Student_Name": "Ali Hassan",      "Roll_No": "ST-102", "Study_Hours": 1.5, "Attendance": 45, "Previous_Marks": 35, "Assignment_Marks": 40},
        {"Student_Name": "Umar Malik",      "Roll_No": "ST-103", "Study_Hours": 5.0, "Attendance": 75, "Previous_Marks": 70, "Assignment_Marks": 68},
        {"Student_Name": "Maryam Siddiqui", "Roll_No": "ST-104", "Study_Hours": 7.0, "Attendance": 92, "Previous_Marks": 89, "Assignment_Marks": 90},
        {"Student_Name": "Hamza Ahmed",     "Roll_No": "ST-105", "Study_Hours": 2.0, "Attendance": 52, "Previous_Marks": 42, "Assignment_Marks": 48},
        {"Student_Name": "Sara Tariq",      "Roll_No": "ST-106", "Study_Hours": 6.0, "Attendance": 85, "Previous_Marks": 78, "Assignment_Marks": 80},
    ])
    output = io.StringIO()
    sample_df.to_csv(output, index=False)
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=sample_student_batch_template.csv"
    response.headers["Content-Type"] = "text/csv"
    return response


@app.route("/batch-prediction/sample/excel")
@login_required
def download_sample_excel():
    sample_df = pd.DataFrame([
        {"Student_Name": "Zainab Fatima",  "Roll_No": "ST-101", "Study_Hours": 6.5, "Attendance": 88, "Previous_Marks": 82, "Assignment_Marks": 85},
        {"Student_Name": "Ali Hassan",      "Roll_No": "ST-102", "Study_Hours": 1.5, "Attendance": 45, "Previous_Marks": 35, "Assignment_Marks": 40},
        {"Student_Name": "Umar Malik",      "Roll_No": "ST-103", "Study_Hours": 5.0, "Attendance": 75, "Previous_Marks": 70, "Assignment_Marks": 68},
        {"Student_Name": "Maryam Siddiqui", "Roll_No": "ST-104", "Study_Hours": 7.0, "Attendance": 92, "Previous_Marks": 89, "Assignment_Marks": 90},
        {"Student_Name": "Hamza Ahmed",     "Roll_No": "ST-105", "Study_Hours": 2.0, "Attendance": 52, "Previous_Marks": 42, "Assignment_Marks": 48},
        {"Student_Name": "Sara Tariq",      "Roll_No": "ST-106", "Study_Hours": 6.0, "Attendance": 85, "Previous_Marks": 78, "Assignment_Marks": 80},
    ])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        sample_df.to_excel(writer, index=False, sheet_name="Batch Template")
    output.seek(0)
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="sample_student_batch_template.xlsx"
    )


# ---------------- ADMIN DASHBOARD ----------------

@app.route("/admin/dashboard")
@login_required
@role_required("admin")
def admin_dashboard():
    student_count = users.count_documents({"role": "student"})
    active_count = users.count_documents({
        "role": "student",
        "active": True
    })
    prediction_count = predictions.count_documents({})
    recent = list(
        predictions.find().sort("created_at", DESCENDING).limit(8)
    )

    return render_template(
        "admin_dashboard.html",
        student_count=student_count,
        active_count=active_count,
        prediction_count=prediction_count,
        recent=recent
    )


@app.route("/admin/users")
@login_required
@role_required("admin")
def admin_users():
    students = list(
        users.find({"role": "student"})
        .sort("created_at", DESCENDING)
    )

    return render_template("admin_users.html", students=students)


@app.route("/admin/users/<user_id>/toggle", methods=["POST"])
@login_required
@role_required("admin")
def toggle_user(user_id):
    try:
        student = users.find_one({
            "_id": ObjectId(user_id),
            "role": "student"
        })
    except (InvalidId, TypeError):
        student = None

    if not student:
        abort(404)

    new_status = not student.get("active", True)

    users.update_one(
        {"_id": student["_id"]},
        {"$set": {"active": new_status}}
    )

    flash("Student account status updated.", "success")
    return redirect(url_for("admin_users"))


@app.route("/admin/history")
@login_required
@role_required("admin")
def admin_history():
    records = list(
        predictions.find()
        .sort("created_at", DESCENDING)
        .limit(200)
    )

    return render_template("admin_history.html", records=records)


# ---------------- PROFILE UPDATE (NAAM FIX) ----------------

@app.route("/profile/update-name", methods=["POST"])
@login_required
def update_profile_name():
    user = current_user()
    new_name = request.form.get("name", "").strip()

    if len(new_name) < 2:
        flash("Naam kam se kam 2 characters ka hona chahiye.", "danger")
        return redirect(request.referrer or url_for("dashboard"))

    try:
        users.update_one(
            {"_id": user["_id"]},
            {"$set": {"name": new_name}}
        )
        flash(f"Naam update ho gaya: {new_name}", "success")
    except PyMongoError:
        app.logger.exception("Profile name update failed.")
        flash("Naam update nahi ho saka.", "danger")

    return redirect(request.referrer or url_for("dashboard"))


# ---------------- ERROR PAGES ----------------

@app.errorhandler(404)
def not_found(error):
    return render_template(
        "error.html",
        code=404,
        message="Page not found."
    ), 404


@app.errorhandler(403)
def forbidden(error):
    return render_template(
        "error.html",
        code=403,
        message="Access denied."
    ), 403


@app.errorhandler(500)
def server_error(error):
    return render_template(
        "error.html",
        code=500,
        message="Something went wrong. Please try again."
    ), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true"
    )
