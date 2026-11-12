import os
from fastapi.concurrency import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import joblib

from contextlib import asynccontextmanager
from src.data_prep import create_attributes

MODEL_PATH = os.getenv("MODEL_PATH", "models/trained_model.pkl")

#Definimos lops dominios de los atributos de entrada
CATEGORIAS = Literal["Alimentos", "Belleza", "Calzado", "Deportes", 
                     "Electrónica", "Hogar", "Indumentaria", "Libros"]
PLATAFORMAS = Literal["Amazon", "Falabela", "Flikart", 
                     "Meesho", "Mercadolibre", "Nykaa", "Zepto"]
CIUDADES = Literal["Buenos Aires", "Córdoba", "La Plata", "Mar del Plata", "Mendoza", 
                   "Posadas", "Rosario", "Salta", "San Juan de Tucumán", "Santa Fe"]
GENEROS = Literal["Femenino", "Masculino", "Otro"]
CANALES = Literal["Online", "Offline"]
SI_NO = Literal["Sí", "No"]


class Resena(BaseModel):
    """Modelo de datos para la reseña|"""
    texto_resena: str = Field(min_length=1, examples=["El envió fue rápido y el parlante llegó en perfectas condiciones.", "No me gustó, esperaba más."])
    categoria_producto: CATEGORIAS = Field(examples=["Electrónica"], description="Categoría del producto")
    canal_compra: CANALES = Field(examples=["Online"], description="Canal de compra del producto")
    plataforma: PLATAFORMAS = Field(examples=["Amazon"], description="Plataforma de compra del producto")
    importe_compra: float = Field(gt=0, examples=[199.99], description="Importe de la compra del producto")
    tiempo_respuesta_horas: float = Field(gt=0, examples=[24.5], description="Tiempo de respuesta del servicio al cliente en horas")
    problema_resuelto: SI_NO = Field(examples=["Sí"], description="Indica si el problema fue resuelto")
    reclamo_formal: SI_NO = Field(examples=["No"], description="Indica si se realizó un reclamo formal")
    #Datos del cliente
    edad: float = Field(ge=0, le=90, examples=[30], description="Edad del usuario")
    genero: GENEROS = Field(examples=["Masculino"], description="Género del usuario")
    ciudad: CIUDADES = Field(examples=["Buenos Aires"], description="Ciudad del usuario")
    antiguedad_cliente_meses: int = Field(ge=0, examples=[12], description="Antigüedad del cliente en meses")

# TODO: incorporar los atriobutos adicionales que hayan sido definidos en el proceso de transformación    

class PredictionResponse(BaseModel):
    """Modelo de datos para la respuesta de la predicción"""
    sentimiento: str 
    probabilidad: dict[str, float] 

modelo = {}

@asynccontextmanager
async def lifespam(app: FastAPI):
    #Cargar el modelo entrenado y el preprocesador al iniciar la aplicación
    if os.path.exists(MODEL_PATH):
        modelo["pipeline"] = joblib.load(MODEL_PATH)
    yield
    modelo.clear()  

app = FastAPI(title="API de predicción de sentimiento",
              description="API para predecir el sentimiento de reseñas de productos",
              version="1.0.0",
              lifespan=lifespam)
@app.get("/")    
def read_root():
    return {"message": "Bienvenido a la API de clasificación de sentimentos. Use el endpoint /predict para obtener las predicciones."}


@app.get("/health")
def health_check():
    """Endpoint para verificar el estado de la API"""
    if "pipeline" in modelo:
        return {"status": "ok", "message": "La API está funcionando correctamente."}
    else:
        raise HTTPException(status_code=503, detail="El modelo no está cargado. La API no está disponible.")


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(resena: Resena):
    """Endpoint para predecir el sentimiento de una reseña"""
    if "pipeline" not in modelo:
        raise HTTPException(status_code=503, detail="El modelo no está cargado. La API no está disponible.")
    
    pipeline = modelo["pipeline"]

    #crear atributos adicionales
    df = create_attributes(pd.DataFrame([resena.dict()]))

    sentimiento = pipeline.predict(df)[0]
    probabilidades = {}

    if hasattr(pipeline, "predict_proba"):
        probabilidades = dict(zip(pipeline.classes_, pipeline.predict_proba(df)[0].round(4).tolist()))
      
    return PredictionResponse(sentimiento=sentimiento, probabilidad=probabilidades)

@app.get("/opciones")
def get_options():
    """Endpoint para obtener las opciones de loa atributos categoricos"""
    return {
        nombre: list(campo.annotation.__args__) 
        for nombre, campo in Resena.model_fields.items() 
        if getattr(campo.annotation, "__origin__",None) is Literal
    }    

