import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Configuración de la app
# -----------------------------
st.set_page_config(page_title="Predicción de Ventas", layout="centered")
st.title("📈 Predicción de Ventas con Red Neuronal")

# -----------------------------
# Datos
# -----------------------------
data = {
    "lluvia": [0.7,13.2,0.1,9,33.5,105,145.6,235.1,159.6,45.5,14.6,3,
               12.3,11.7,8.4,2.5,64.6,211.6,131.4,173.7,196.7,71,1.6,8.7],

    "temperatura": [16.4,17.4,20.3,21,23.6,27.5,19.2,19.4,19.3,16.9,17.4,15.5,
                    15.9,17.5,19.8,20.2,21.6,19.2,18.7,19,19.3,17.9,16.3,15.5],

    "clasificador": [2,2,1,0,0,0,2,2,1,0,1,0,
                     2,2,1,0,0,0,2,2,1,0,1,0],

    "ventas": [10240,11197,7014,11620,15239,9549,6619,12156,3796,
               4435.5,4409,2731,7738.7,7167,1441.65,8285.48,4097,
               9721.5,6579.42,8502,3796,2494,279,3322.43]
}

df = pd.DataFrame(data)

st.subheader("📊 Datos")
st.dataframe(df)

# -----------------------------
# Variables
# -----------------------------
X = df[["lluvia", "temperatura", "clasificador"]]
y = df["ventas"]

# Escalamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# Modelo de Red Neuronal
# -----------------------------
model = Sequential([
    Dense(16, activation="relu", input_shape=(3,)),
    Dense(8, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss="mse"
)

# -----------------------------
# Entrenamiento
# -----------------------------
st.subheader("⚙️ Entrenamiento")

if st.button("Entrenar modelo"):
    history = model.fit(
        X_train, y_train,
        epochs=200,
        verbose=0,
        validation_split=0.2
    )

    st.success("Modelo entrenado ✅")

    # Evaluación
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**R²:** {r2:.4f}")

    # Gráfica
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Real")
    ax.set_ylabel("Predicho")
    ax.set_title("Real vs Predicción")
    st.pyplot(fig)

# -----------------------------
# Predicción interactiva
# -----------------------------
st.subheader("🔮 Predicción")

lluvia = st.number_input("Lluvia", value=10.0)
temperatura = st.number_input("Temperatura", value=20.0)
clasificador = st.selectbox("Clasificador", [0, 1, 2])

if st.button("Predecir"):
    entrada = np.array([[lluvia, temperatura, clasificador]])
    entrada_scaled = scaler.transform(entrada)

    pred = model.predict(entrada_scaled)

    st.success(f"💰 Ventas estimadas: {pred[0][0]:.2f}")
