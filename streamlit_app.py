import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Force Streamlit to redeploy

st.set_page_config(page_title="Age Inference Tool", page_icon="🧠", layout="centered")

# 🌸 Pastel Dreamy Styling
st.markdown("""
<style>
/* General Page Styling */
body {
    background-color: #fff7fb;
    font-family: 'Segoe UI', sans-serif;
}

html, .block-container {
    padding: 2rem;
    background-color: #fff7fb;
}

/* Title */
h1 {
    color: #6a1b9a;
}

/* Button Styling */
.stButton > button {
    background-color: #f8bbd0;
    color: #4a148c;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    padding: 0.5em 1.5em;
    transition: 0.3s;
}
.stButton > button:hover {
    background-color: #f48fb1;
}

/* Download Button */
.stDownloadButton > button {
    background-color: #ba68c8;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    padding: 0.5em 1.5em;
    transition: 0.3s;
}
.stDownloadButton > button:hover {
    background-color: #ab47bc;
}

/* Input boxes (including selectbox, file upload) */
.stTextInput, .stSelectbox, .stFileUploader {
    background-color: #f3e5f5 !important;
    border-radius: 10px !important;
    padding: 12px !important;
    color: #4a148c !important;
}

/* Text Area (FINALLY get this one right) */
textarea {
    background-color: #f3e5f5 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    color: #4a148c !important;
    border: 1px solid #ce93d8 !important;
    box-shadow: none !important;
    width: 100% !important;
}

/* Confidence box */
div[data-testid="stMarkdownContainer"] > div {
    font-size: 1.1em;
    line-height: 1.6em;
}

/* Table styling */
.css-1d391kg, .stDataFrame {
    background-color: #fce4ec;
    border-radius: 12px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)



# ✅ Page title and intro message
st.title("🧠 On / Off Age Group Inference Tool")
st.markdown("""
Predict the age group of a message based on its style, tone, and content. All predictions are guesses based on synthetic data and should not be treated as definitive. 
Choose a platform below and try it out with a single message or upload a CSV for batch predictions.
""")
st.markdown("---")

# ✅ Platform selection with a cleaner layout
st.markdown("### 🔍 Select a Platform")
platform = st.selectbox("", ["Roblox", "TikTok", "Character.ai"])

# Mode selector
mode = st.radio("Choose prediction mode:", ["Single message", "User-level (multiple messages)"])

# ✅ Friendly user instruction with soft background
st.markdown("""
<div style="background-color:#f0f2f6; padding:10px; border-radius:8px;">
👤 You can either enter a message below or upload a CSV for batch predictions.
</div>
""", unsafe_allow_html=True)

# Load the appropriate model and vectorizer based on selection
if platform == "Roblox":
    model = joblib.load("roblox_classifier.pkl")
    vectorizer = joblib.load("roblox_vectorizer.pkl")
    label_names = ["Adult (18+)", "Teen (13–17)", "Child (0–12)"]
elif platform == "TikTok":
    model = joblib.load("tiktok_classifier.pkl")
    vectorizer = joblib.load("tiktok_vectorizer.pkl")
    label_names = ["Not Minor (18+)", "Minor (under 18)"]
elif platform == "Character.ai":
    model = joblib.load("characterai_classifier.pkl")
    vectorizer = joblib.load("characterai_vectorizer.pkl")
    label_names = ["Adult", "Minor"]

# 🧠 SINGLE MESSAGE MODE
if mode == "Single message":
    user_text = st.text_area("Enter a message:", height=100, key="single_message_input")

    if st.button("Predict Message", key="predict_single"):
        if not user_text.strip():
            st.warning("Please enter a message.")
        else:
            X = vectorizer.transform([user_text])
            pred = int(model.predict(X)[0])
            probs = model.predict_proba(X)[0]

            label = label_names[pred]
            confidence = "  |  ".join(
                f"{label_names[i]}: {round(probs[i] * 100, 1)}%" for i in range(len(label_names))
            )

            st.markdown(f"""
            <div style="background-color:#e0f7fa; padding:10px; border-radius:8px; border-left: 5px solid #00acc1;">
            <b>✅ Predicted:</b> <span style="font-size: 1.2em;">{label}</span><br>
            <b>Confidence:</b> {confidence}
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("📋 Paste multiple messages from a single user (one per line):")
    multi_message_input = st.text_area("User's messages", height=200, key="user_level_input")

    if st.button("Predict User Age", key="predict_user"):
        messages = [m.strip() for m in multi_message_input.strip().split("\n") if m.strip()]
        if not messages:
            st.warning("Please enter at least one message.")
        else:
            X = vectorizer.transform(messages)
            probs = model.predict_proba(X)
            avg_probs = probs.mean(axis=0)
            pred = int(avg_probs.argmax())

            label = label_names[pred]
            confidence = "\n".join(
                [f"- {label_names[i]}: {round(avg_probs[i]*100, 2)}%" for i in range(len(label_names))]
            )

            st.markdown(f"""
            <div style="background-color:#f3e5f5; padding:15px; border-radius:10px; border-left: 6px solid #8e24aa;">
            🧠 <b>Predicted age group for this user:</b> <span style="font-size: 1.3em;">{label}</span><br>
            📈 <b>Based on {len(messages)} messages</b><br><br>
            <b>Confidence Breakdown:</b><br>
            {confidence}
            </div>
            """, unsafe_allow_html=True)


# Optional batch prediction
st.markdown("---")
st.subheader("📁 Batch Prediction (CSV)")
uploaded_file = st.file_uploader("Upload a CSV file with a 'message' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "message" not in df.columns:
        st.error("CSV must have a 'message' column.")
    else:
        X = vectorizer.transform(df["message"].astype(str))
        preds = model.predict(X)
        probas = model.predict_proba(X)

        df["predicted_label"] = [label_names[int(p)] for p in preds]
        df["confidence"] = [
            " / ".join([f"{label_names[i]} {round(prob[i] * 100, 1)}%" for i in range(len(label_names))])
            for prob in probas
        ]

        st.dataframe(df[["message", "predicted_label", "confidence"]])
        st.download_button("Download Results as CSV", df.to_csv(index=False), "predictions_output.csv", "text/csv")

