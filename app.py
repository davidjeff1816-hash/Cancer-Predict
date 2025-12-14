import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------
# APP TITLE
# -------------------------
st.title("🩺 Cancer Prediction App")
st.write("Logistic Regression based cancer prediction")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("The_Cancer_data_1500_V2.csv")
    return data

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------
# FEATURE SELECTION
# -------------------------
X = df[['age', 'mass', 'insu', 'plas']]
y = df['class']

# -------------------------
# TRAIN MODEL
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------
# MODEL ACCURACY
# -------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"Model Accuracy: {accuracy:.2f}")

# -------------------------
# USER INPUT
# -------------------------
st.sidebar.header("Enter Patient Details")

age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
mass = st.sidebar.number_input("Mass", min_value=0.0, value=25.0)
insu = st.sidebar.number_input("Insulin Level", min_value=0.0, value=80.0)
plas = st.sidebar.number_input("Plasma Level", min_value=0.0, value=120.0)

# -------------------------
# PREDICTION
# -------------------------
if st.sidebar.button("Predict"):
    input_data = [[age, mass, insu, plas]]
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Cancer Detected")
    else:
        st.success("✅ No Cancer Detected")
