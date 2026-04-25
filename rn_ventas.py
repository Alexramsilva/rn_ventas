import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Configuración
st.set_page_config(page_title="RN Ventas Mejorada", page_icon="📈")
st.title("📈 Predicción de Ventas (Red Neuronal Mejorada)")

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
st.dataframe(df)

# -----------------------------
# Variables
# -----------------------------
X = df[["lluvia", "temperatura", "clasificador"]]
y = df["ventas"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# Modelo mejorado
# -----------------------------
model = Sequential([
    Input(shape=(3,)),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse"
)

# Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

# Entrenamiento
history = model.fit(
    X_train, y_train,
    epochs=300,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=0
)

st.success("Modelo entrenado")

# -----------------------------
# Evaluación
# -----------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"MSE: {mse:.2f}")
st.write(f"R²: {r2:.4f}")

# -----------------------------
# Gráfica de pérdida
# -----------------------------
fig2, ax2 = plt.subplots()
ax2.plot(history.history["loss"], label="Entrenamiento")
ax2.plot(history.history["val_loss"], label="Validación")
ax2.set_title("Evolución de la pérdida")
ax2.legend()

st.pyplot(fig2)

# -----------------------------
# Predicción interactiva
# -----------------------------
st.subheader("🔮 Nueva predicción")

lluvia = st.slider("Lluvia", float(df.lluvia.min()), float(df.lluvia.max()), 10.0)
temperatura = st.slider("Temperatura", float(df.temperatura.min()), float(df.temperatura.max()), 20.0)
clasificador = st.selectbox("Clasificador", [0,1,2])

nuevo = np.array([[lluvia, temperatura, clasificador]])
nuevo_scaled = scaler.transform(nuevo)

pred = model.predict(nuevo_scaled)
st.write(f"Ventas estimadas: {pred[0][0]:.2f}")
