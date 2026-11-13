"""
    Interface de usuario para la API de clasificación de sentimientos
"""    

import os
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

COLORES = {"Positivo": "🟢", "Neutral": "🟡", "Negativo": "🔴"}

st.set_page_config(page_title="Sentimientos en reseñas", layout="wide")
st.title("Clasificación de sentimientos en reseñas de productos")
#TODO: completar nombres con los autores del trabajos
st.caption("[autor/er] - Proyecto 13MBID - Ingrese los datos de la reseña y obtenga la predicción del sentimiento.")

@st.cache_data(ttl=300)
def obtener_opciones():
    """
    Función para obtener las opciones de categorías de productos y ciudades desde la API
    Returns:
        dict: Diccionario con las opciones de categorías y ciudades
    """
    try:
        response = requests.get(f"{API_URL}/opciones", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error al obtener las opciones de la API: {e}")
        st.stop()

try:
    opciones = obtener_opciones()
except requests.exceptions.RequestException:
    st.error(f"No se pudo conectar con la API. Por favor, asegúrese de que la API este en funcionamiento")
    st.stop()

with st.form("formulario_reseña"):
    st.subheader("Ingrese los datos de la reseña y la compra")
    texto_reseña = st.text_area("Texto de la reseña", height=150, placeholder="Ingrese el texto de la reseña aquí...")
    col1, col2 = st.columns(2)
    categoria_producto = col1.selectbox("Categoría del producto", opciones["categoria_producto"])
    plataforma= col2.selectbox("Plataforma de compra", opciones["plataforma"])
    canal_compra = col1.selectbox("Canal de compra", opciones["canal_compra"])
    importe_compra = col2.number_input("Importe de la compra", min_value=1.0, step=500.0, value=25000.0, format="%.2f")

    st.subheader("Atención postventa")
    col3, col4, col5 = st.columns(3)
    tiempo_respuesta_horas = col3.number_input("Tiempo de respuesta del servicio al cliente (horas)", min_value=0.1, step=1.0, value=24.0, format="%.2f")
    problema_resuelto = col4.radio("¿El problema fue resuelto?", opciones["problema_resuelto"], horizontal=True)
    reclamo_formal = col5.radio("¿Se realizó un reclamo formal?", opciones["reclamo_formal"], index=1, horizontal=True)

    st.subheader("Datos del cliente")
    col6, col7 = st.columns(2)
    edad = col6.number_input("Edad del usuario", min_value=18, max_value=90, step=1, value=30)
    genero = col7.selectbox("Género del usuario", opciones["genero"])  
    ciudad = col6.selectbox("Ciudad del usuario", opciones["ciudad"])
    antiguedad_cliente_meses = col7.number_input("Antigüedad del cliente (meses)", min_value=0, step=1, value=12)

    enviar = st.form_submit_button("Predecir sentimiento", type="primary")

    if enviar:
        if not texto_reseña.strip():
            st.error("El campo 'texto de la reseña' no puede estar vacío.")

        datos = {
            "texto_resena": texto_reseña,
            "categoria_producto": categoria_producto,
            "canal_compra": canal_compra,
            "plataforma": plataforma,
            "importe_compra": importe_compra,
            "tiempo_respuesta_horas": tiempo_respuesta_horas,
            "problema_resuelto": problema_resuelto,
            "reclamo_formal": reclamo_formal,
            "edad": edad,
            "genero": genero,
            "ciudad": ciudad,
            "antiguedad_cliente_meses": antiguedad_cliente_meses
        }

        try:
            respuesta = requests.post(f"{API_URL}/predict", json=datos, timeout=30)
        except requests.exceptions.RequestException:
            st.error("Error al conectar con la API. Por favor, intente nuevamente más tarde.")
            st.stop()

        if respuesta.status_code != 200:
            st.error(f"Error en la predicción: {respuesta.status_code} - {respuesta.text}")
            st.stop()

        resultado = respuesta.json()
        sentimiento = resultado.get("sentimiento")
        st.metric("Sentimiento estimado", f"{COLORES.get(sentimiento, '#000000')} {sentimiento}")    

        if "probabilidad" in resultado:
            probabilidades = pd.Series(resultado["probabilidad"], name="Probabilidad")
            st.bar_chart(probabilidades)
            if sentimiento == "Negativo":
                st.warning("El sentimiento de la reseña es negativo. Se recomienda revisar la atención al cliente y el producto.")

###########################################################################
#  TODO: agregar los valores o campos que sean necesarios para el calculo de otros
# atributos en caso de corresponder
###########################################################################





        


