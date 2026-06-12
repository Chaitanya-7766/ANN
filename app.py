import streamlit as st
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="ANN - Iris Classifier", layout="centered")
st.title("🧠 Artificial Neural Network (ANN)")
st.markdown("Classify Iris flowers using a simple feedforward ANN built with Keras.")

data = load_iris()
X, y = data.data, data.target
feature_names = [f.replace(" (cm)", "") for f in data.feature_names]
target_names = data.target_names

# ----------------------------------------------------------------------
# 1. Dataset Preview
# ----------------------------------------------------------------------
st.header("📊 Dataset Preview")
df = pd.DataFrame(X, columns=feature_names)
df["species"] = [target_names[i] for i in y]
st.dataframe(df.head(10), use_container_width=True)
st.markdown("**Class distribution:**")
st.bar_chart(df["species"].value_counts())

# ----------------------------------------------------------------------
# 2. Model Training
# ----------------------------------------------------------------------
st.header("🏋️ Model Training")


@st.cache_resource
def train_model():
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)

    model = keras.Sequential([
        keras.layers.Dense(16, activation="relu", input_shape=(4,)),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=50, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    return model, scaler, acc


with st.spinner("Training ANN..."):
    model, scaler, acc = train_model()

st.success(f"Model trained successfully! Test Accuracy: {acc * 100:.2f}%")

# ----------------------------------------------------------------------
# 3. Output - Old Data vs New Data Comparison
# ----------------------------------------------------------------------
st.header("🔮 Output")
st.markdown("Compare a prediction on an **existing flower from the dataset** (old data) with a prediction on **your own custom flower** (new data).")

idx = st.slider("Pick a flower index from the dataset:", 0, len(X) - 1, 0)
old_features = X[idx]

st.markdown("**Enter measurements for a new flower:**")
c1, c2, c3, c4 = st.columns(4)
with c1:
    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8)
with c2:
    sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
with c3:
    petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
with c4:
    petal_width = st.slider("Petal Width", 0.1, 2.5, 1.2)
new_features = np.array([sepal_length, sepal_width, petal_length, petal_width])

if st.button("Compare Predictions"):
    old_pred = model.predict(scaler.transform([old_features]), verbose=0)[0]
    new_pred = model.predict(scaler.transform([new_features]), verbose=0)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📁 Old Data (Dataset Sample)")
        st.dataframe(pd.DataFrame([old_features], columns=feature_names), use_container_width=True)
        st.write(f"**True Species:** {target_names[y[idx]]}")
        st.write(f"**Predicted Species:** {target_names[np.argmax(old_pred)]}")
        st.bar_chart(pd.DataFrame({"Probability": old_pred}, index=target_names))
    with col2:
        st.subheader("🆕 New Data (Your Input)")
        st.dataframe(pd.DataFrame([new_features], columns=feature_names), use_container_width=True)
        st.write(f"**Predicted Species:** {target_names[np.argmax(new_pred)]}")
        st.bar_chart(pd.DataFrame({"Probability": new_pred}, index=target_names))
