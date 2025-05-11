import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import numpy as np
import time


st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection")

st.sidebar.header("⬆️ Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.subheader("📄 Dataset Preview")
    st.write(df.head())

    st.subheader("📊 Class Distribution")
    class_counts = df['Class'].value_counts()
    st.bar_chart(class_counts)

    class_counts = df['Class'].value_counts()
    total = class_counts.sum()
    fraud = class_counts.get(1, 0)
    legit = class_counts.get(0, 0)
    fraud_pct = (fraud / total) * 100
    legit_pct = (legit / total) * 100

    st.markdown(f"""
    - 🧾 **Total Transactions:** {total:,}
    - ✅ **Legitimate Transactions:** {legit:,} ({legit_pct:.2f}%)
    - ⚠️ **Fraudulent Transactions:** {fraud:,} ({fraud_pct:.2f}%)
    """)

    st.subheader("📋 Exploratory Data Analysis")

    st.markdown("#### Transaction Amount Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Amount'], bins=50, ax=ax1)
    st.pyplot(fig1)

    st.markdown("#### Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.subheader("🎓 Model Training & Results (Logistic Regression)")
    progress = st.progress(0, text="🚀 Initializing training...")

    X = df.drop(columns=['Class'])
    y = df['Class']
    time.sleep(1)
    progress.progress(10, "🔄 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    time.sleep(1)
    progress.progress(30, "✅ Features scaled.")

    time.sleep(1)
    progress.progress(40, "📂 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    time.sleep(1)
    progress.progress(60, "✅ Dataset split.")

    time.sleep(1)
    progress.progress(70, "🚀 Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    time.sleep(1)
    progress.progress(85, "✅ Model trained.")

    time.sleep(1)
    progress.progress(90, "🧪 Generating predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    time.sleep(1)
    progress.progress(100, "✅ Done!")

    st.markdown("#### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig3, ax3 = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax3)
    st.pyplot(fig3)

    st.markdown("#### ROC AUC Score")
    st.metric(label="ROC AUC", value=f"{roc_auc_score(y_test, y_proba):.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    st.markdown("#### ROC Curve")
    fig4, ax4 = plt.subplots()
    ax4.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.4f}")
    ax4.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax4.set_xlabel("False Positive Rate")
    ax4.set_ylabel("True Positive Rate")
    ax4.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax4.legend(loc="lower right")
    st.pyplot(fig4)

    st.subheader("🔍 Predict a Single Transaction")

    with st.form("prediction_form"):
        st.markdown("Enter values for each feature (auto-filled with sample values):")

        feature_names = list(X.columns)
        input_data = []
        sample_row = X.iloc[0].tolist()

        for i, fname in enumerate(feature_names):
            val = st.number_input(f"{fname}", value=float(sample_row[i]), step=0.01, key=f"feature_{i}")
            input_data.append(val)

        submit = st.form_submit_button("Predict")

        if submit:
            st.info("🔄 Running prediction...")
            time.sleep(1)
            input_array = pd.DataFrame([input_data], columns=feature_names)
            input_array_scaled = scaler.transform(input_array)
            pred = model.predict(input_array_scaled)[0]
            proba = model.predict_proba(input_array_scaled)[0][1]

            if pred == 1:
                st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {proba:.2%})")
            else:
                st.success(f"✅ Legitimate Transaction. (Probability: {proba:.2%})")

else:
    st.info("👈 Upload a dataset in the sidebar to get started.")

st.markdown("---")
st.markdown(
    """
    💻**Developed by VIJAY SHANKAR (1CR22IS185)**  
    🎓 BE in Information Science  
    📝 Project: Credit Card Fraud Detection  
    📧 vish22ise@cmrit.ac.in  
    """
)
