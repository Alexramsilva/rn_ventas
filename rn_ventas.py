# Crear modelo solo una vez
if "model" not in st.session_state:
    st.session_state.model = MLPRegressor(
        hidden_layer_sizes=(16, 8),
        max_iter=2000,
        random_state=42
    )

model = st.session_state.model
