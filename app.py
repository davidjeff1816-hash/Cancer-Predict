import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Cancer Prediction App", layout="wide")

st.title("🩺 Cancer Prediction App")
st.write("Logistic Regression based Cancer Prediction")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("The_Cancer_data_1500_V2.csv")
    df.columns = df.columns.str.lower().str.strip()
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------
# FEATURES & TARGET (MATCH DATASET)
# -------------------------
features = [
    'age',
    'bmi',
    'physicalactivity',
    'alcoholintake'
]

target = 'diagnosis'

X = df[features]
y = df[target]

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
accuracy = accuracy_score(y_test, model.predict(X_test))
st.success(f"Model Accuracy: {accuracy:.2f}")

# -------------------------
# SIDEBAR INPUT
# -------------------------
st.sidebar.header("Enter Patient Details")

age = st.sidebar.number_input("Age", 0, 120, 30)
bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0)
physicalactivity = st.sidebar.number_input("Physical Activity Level", 0.0, 20.0, 5.0)
alcoholintake = st.sidebar.number_input("Alcohol Intake Level", 0.0, 20.0, 4.0)

# -------------------------
# PREDICTION
# -------------------------
if st.sidebar.button("Predict"):
    input_df = pd.DataFrame(
        [[age, bmi, physicalactivity, alcoholintake]],
        columns=features
    )

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Cancer Detected")
    else:
        st.success("✅ No Cancer Detected")
