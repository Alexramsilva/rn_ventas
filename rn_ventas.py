import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Configuración de la página
st.set_page_config(page_title="Predicción de Ventas con RN", layout="centered")

st.title("📊 Predicción de Ventas con Red Neuronal")

# Cargar archivo
archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)

    st.subheader("Vista previa de datos")
    st.write(df.head())

    # Selección de variables
    columnas = df.columns.tolist()
    target = st.selectbox("Selecciona la variable objetivo (Y)", columnas)
    features = st.multiselect("Selecciona variables predictoras (X)", [c for c in columnas if c != target])

    if len(features) > 0:
        X = df[features].values
        y = df[target].values

        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Escalamiento
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Modelo
        model = Sequential([
            Dense(16, activation="relu", input_shape=(X_train_scaled.shape[1],)),
            Dropout(0.2),
            Dense(8, activation="relu"),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

        # Entrenamiento
        if st.button("Entrenar modelo"):
            early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

            history = model.fit(
                X_train_scaled, y_train,
                validation_split=0.2,
                epochs=200,
                batch_size=16,
                verbose=0,
                callbacks=[early_stop]
            )

            st.success("Modelo entrenado correctamente ✅")

            # Evaluación
            y_pred = model.predict(X_test_scaled)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write(f"MSE: {mse:.4f}")
            st.write(f"R²: {r2:.4f}")

            # Gráfica
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Valores reales")
            ax.set_ylabel("Predicciones")
            ax.set_title("Real vs Predicción")
            st.pyplot(fig)

            # Guardar en sesión
            st.session_state["model"] = model
            st.session_state["scaler"] = scaler
            st.session_state["features"] = features

        # Predicción manual
        if "model" in st.session_state:
            st.subheader("🔮 Nueva predicción")

            entrada = []
            for col in st.session_state["features"]:
                val = st.number_input(f"Ingrese {col}", value=0.0)
                entrada.append(val)

            if st.button("Predecir"):
                entrada = np.array(entrada).reshape(1, -1)
                entrada_scaled = st.session_state["scaler"].transform(entrada)

                pred = st.session_state["model"].predict(entrada_scaled)
                st.success(f"Predicción: {pred[0][0]:.4f}")

else:
    st.info("Por favor, sube un archivo CSV para comenzar.")
