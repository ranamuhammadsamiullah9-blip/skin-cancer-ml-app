import os
import glob
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Skin Cancer Detector", layout="centered")
st.title("Skin Cancer Detector")

# 1. CSV file automatically locate karein
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)

if not csv_files:
    st.error("HAM10000_metadata.csv file repository mein nahi mili.")
    st.stop()

df = pd.read_csv(csv_files[0])

# Target column ('dx'), irrelevant ID columns aur 'localization' remove karein
target_col = "dx" if "dx" in df.columns else df.columns[-1]
drop_cols = [
    c for c in ["lesion_id", "image_id", "localization", target_col]
    if c in df.columns
]

# Data clean aur encode karein
clean_df = df.dropna(subset=[target_col]).copy()
feature_cols = [c for c in clean_df.columns if c not in drop_cols]

X_raw = clean_df[feature_cols].copy()
for col in X_raw.select_dtypes(include=["object", "category"]).columns:
    X_raw[col] = X_raw[col].fillna("unknown").astype(str)
for col in X_raw.select_dtypes(exclude=["object", "category"]).columns:
    X_raw[col] = X_raw[col].fillna(X_raw[col].median())

X = pd.get_dummies(X_raw, drop_first=False)
y = clean_df[target_col].astype(str)

@st.cache_resource
def train_detector(features_df, target_series):
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, target_series, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, acc

model, acc = train_detector(X, y)

# 2. Skin Cancer Detector Box
with st.container(border=True):
    st.subheader("Patient & Lesion Details")
    st.caption(f"Model Accuracy on Metadata: {acc * 100:.2f}%")

    # Patient Name input (dx ki jagah)
    patient_name = st.text_input("Patient Name", placeholder="Enter patient full name")

    # Image upload box
    uploaded_image = st.file_uploader(
        "Upload Skin Lesion Image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Skin Image", use_container_width=True)

    user_input = {}

    for col in feature_cols:
        if col not in clean_df.columns:
            continue
        series = clean_df[col]
        if pd.api.types.is_numeric_dtype(series):
            default_val = float(series.median()) if not series.dropna().empty else 0.0
            min_val = float(series.min()) if not series.dropna().empty else 0.0
            max_val = float(series.max()) if not series.dropna().empty else 100.0
            user_input[col] = st.number_input(
                label=col.replace("_", " ").title(),
                min_value=min_val,
                max_value=max_val,
                value=default_val
            )
        else:
            options = sorted(series.dropna().astype(str).unique().tolist())
            if not options:
                options = ["unknown"]
            user_input[col] = st.selectbox(
                label=col.replace("_", " ").title(),
                options=options
            )

    predict_clicked = st.button("Detect Skin Condition", type="primary", use_container_width=True)

    if predict_clicked:
        input_row_df = pd.DataFrame([user_input])
        for col in input_row_df.select_dtypes(include=["object", "category"]).columns:
            input_row_df[col] = input_row_df[col].astype(str)

        input_encoded = pd.get_dummies(input_row_df, drop_first=False)
        input_aligned = input_encoded.reindex(columns=X.columns, fill_value=0)

        prediction = model.predict(input_aligned)[0]
        probs = model.predict_proba(input_aligned)[0]
        confidence = max(probs) * 100

        display_name = patient_name.strip() if patient_name.strip() else "Unnamed Patient"

        st.divider()
        st.success(f"**Patient Name:** {display_name}")
        st.info(f"**Detected Result:** `{prediction}`")
        st.write(f"**Confidence:** {confidence:.2f}%")
