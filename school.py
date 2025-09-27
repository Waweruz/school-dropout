import streamlit as st
import pandas as pd
import numpy as np

# --- Check imports for scikit-learn ---
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
except ImportError as e:
    st.error("⚠️ scikit-learn is not installed. Please make sure 'scikit-learn' "
             "is included in your requirements.txt at the project root.")
    raise e

# --- App title ---
st.title("🎓 School Dropout Predictor")

# --- Upload dataset ---
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.write("📊 Preview of Dataset:")
    st.dataframe(data.head())

    # --- Basic preprocessing ---
    st.subheader("Preprocessing")
    st.write("Please select the target column (the one to predict, e.g. 'Dropout').")
    target_col = st.selectbox("Target Column", data.columns)

    feature_cols = [c for c in data.columns if c != target_col]
    X = data[feature_cols]
    y = data[target_col]

    # Handle non-numeric columns
    X = pd.get_dummies(X, drop_first=True)

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Model training ---
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # --- Predictions ---
    y_pred = model.predict(X_test)

    # --- Results ---
    st.subheader("Model Performance")
    acc = accuracy_score(y_test, y_pred)
    st.write(f"✅ Accuracy: **{acc:.2f}**")

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # --- Try new prediction ---
    st.subheader("Try Prediction")
    user_input = {}
    for col in feature_cols:
        val = st.text_input(f"Enter value for {col}", "")
        user_input[col] = val

    if st.button("Predict Dropout"):
        try:
            user_df = pd.DataFrame([user_input])
            user_df = pd.get_dummies(user_df)
            # align with training columns
            user_df = user_df.reindex(columns=X.columns, fill_value=0)

            prediction = model.predict(user_df)[0]
            st.success(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
