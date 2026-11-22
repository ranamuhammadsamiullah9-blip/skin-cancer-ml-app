import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(page_title="Skin Cancer ML App", layout="centered")
st.title("Skin Cancer Prediction App")

# 1. Automatic CSV File Finder (FileNotFoundError se bachne ke liye)
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)

df = None

if csv_files:
    # Agar GitHub repo mein koi CSV file maujood hai to pehli CSV load karega
    dfp = csv_files[0]
    df = pd.read_csv(dfp)
    st.success(f"Dataset loaded successfully: `{os.path.basename(dfp)}`")
else:
    # Agar GitHub repo mein CSV upload nahi hui, to manual upload ka option dega
    st.warning("Repository mein koi CSV file nahi mili. Neeche apni dataset CSV file upload karein:")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

# 2. Data Processing & Model Training
if df is not None:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Target column select karein (default aakhri column hoga)
    target_col = st.selectbox(
        "Target Column (Label) select karein:",
        options=df.columns,
        index=len(df.columns) - 1
    )

    # Sirf numeric features aur non-null rows lein
    data = df.dropna().copy()
    X = pd.get_dummies(data.drop(columns=[target_col]), drop_first=True)
    y = data[target_col]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Accuracy Display
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.metric(label="Model Accuracy", value=f"{acc * 100:.2f}%")

    # 3. Prediction Form
    st.subheader("New Prediction")
    input_data = {}
    for col in X.columns:
        val = float(X[col].median())
        input_data[col] = st.number_input(f"{col}", value=val)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.info(f"Predicted Result: **{prediction}**")
