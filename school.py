# app.py
import streamlit as st
import pandas as pd
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("school drop out.xlsx")
    # Handle missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Encode categorical
    encoder = LabelEncoder()
    for column in df.select_dtypes(include="object").columns:
        df[column] = encoder.fit_transform(df[column])

    # Scale features
    scaler = StandardScaler()
    df[df.select_dtypes(include="number").columns] = scaler.fit_transform(
        df.select_dtypes(include="number")
    )
    return df

df = load_data()

# -------------------------------
# Train Models
# -------------------------------
X = df.drop("dropout", axis=1)
y = df["dropout"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine (SVM)": SVC(kernel="linear", random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1-Score": report["1"]["f1-score"],
    }

results_df = pd.DataFrame(results).T

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎓 Student Dropout Predictor (Streamlit Version)")
st.write("This app predicts student dropout likelihood using ML models and stores results in a database.")

# Show dataset
if st.checkbox("Show Dataset"):
    st.dataframe(df.head())

# Show model evaluation
if st.checkbox("Show Model Performance"):
    st.dataframe(results_df)

# -------------------------------
# Prediction Form
# -------------------------------
st.subheader("📊 Enter Student Data for Prediction")

with st.form("student_form"):
    student_id = st.number_input("Student ID", min_value=1, step=1)
    satisfaction = st.slider("School Satisfaction (1-5)", 1, 5, 3)
    attendance = st.slider("Attendance Rate (%)", 1, 100, 70)
    failed_courses = st.slider("Failed Courses", 0, 10, 0)
    commute = st.slider("Commute Time (minutes)", 1, 120, 30)
    disciplinary = st.slider("Disciplinary Incidents", 0, 10, 0)
    homework = st.slider("Homework Completion (%)", 0, 100, 80)
    family_income = st.selectbox("Family Income", ["low", "medium", "high"])

    model_choice = st.selectbox("Choose Model", list(models.keys()))

    submitted = st.form_submit_button("Predict")

# -------------------------------
# Database Setup
# -------------------------------
def init_db():
    conn = sqlite3.connect("students.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS students (
        student_id INTEGER PRIMARY KEY,
        school_satisfaction FLOAT,
        attendance_rate FLOAT,
        failed_courses INTEGER,
        commute_time INTEGER,
        disciplinary_incidents INTEGER,
        homework_completion FLOAT,
        family_income TEXT,
        promotion_status TEXT
    );''')
    conn.commit()
    return conn

conn = init_db()

if submitted:
    # Simple promotion logic
    promotion_status = (
        "Promoted"
        if (satisfaction > 3 and attendance > 70 and failed_courses <= 2
            and commute <= 40 and disciplinary <= 2 and homework > 80)
        else "Dropped Out"
    )

    # Save to DB
    conn.execute(
        "INSERT OR REPLACE INTO students VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (student_id, satisfaction, attendance, failed_courses, commute,
         disciplinary, homework, family_income, promotion_status),
    )
    conn.commit()

    st.success(f"✅ Prediction: Student {student_id} is **{promotion_status}**")

# -------------------------------
# View Database
# -------------------------------
if st.checkbox("View Database Records"):
    db_df = pd.read_sql("SELECT * FROM students", conn)
    st.dataframe(db_df)
