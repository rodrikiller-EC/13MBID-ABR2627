# Se importan las librerías necesarias y se suprimen las advertencias
from flask.cli import load_dotenv
from mlflow.models import infer_signature
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from dotenv import load_dotenv, find_dotenv
import mlflow
import mlflow.sklearn
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=UserWarning)

REGISTERED_MODEL_NAME = "sentimiento-resenas"
ALIAS_PRODUCCION = "champion"
UMBRAL_ACCURACY = 0.8


def load_data(data_path: str = "data/processed/datos_integrados.csv"):
    """
    Carga los datos desde un archivo CSV y devuelve un DataFrame de pandas.
    
    Args:
        data_path (str): Ruta al archivo CSV que contiene los datos.
        
    Returns:
        pd.DataFrame: DataFrame que contiene los datos cargados.
    """
    #lectura de los datos
    df = pd.read_csv(data_path)

    target = "sentimiento"
    features_X =df.drop(columns=[target])
    labels_y = df[target]

    # Primero se separa el test final (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        features_X,
        labels_y,
        test_size=0.10,
        random_state=42,
        stratify=labels_y
    )

    # Luego se separan train y validation (22% del 90% es aprox. el 20% del total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.22,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, features_X, labels_y

def create_preprocessor(features_X: pd.DataFrame):
    """
    Crea un preprocesador para los datos de entrada.
    
    Args:
        features_X (pd.DataFrame): DataFrame que contiene las características de entrada.
        
    Returns:
        ColumnTransformer: Preprocesador que aplica transformaciones a las columnas especificadas.
    """
    # Identificar las columnas numéricas y categóricas
    numeric_features = features_X.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()
    categorical_features = features_X.select_dtypes(
        include=['object', 'string']
    ).columns.tolist()

    # Definición de transformadores para cada tipo de característica
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    text_col = "texto_resena"
    text_transformer = TfidfVectorizer(max_features=300)

    # Crear el preprocesador utilizando ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('text', text_transformer, text_col)
        ]
    )

    return preprocessor

def train_model(
    data_path: str = "data/processed/datos_integrados.csv",
    model_output_path: str = "models/trained_model.pkl",
    preprocessor_output_path: str = "models/preprocessor.pkl",
    metrics_output_path: str = "metrics/train_metrics.json",
    figures_output_path: str = "docs/figures",
    register_model_name: str | None = REGISTERED_MODEL_NAME,
):

    """
    Entrena un modelo de clasificación utilizando los datos proporcionados y guarda el modelo entrenado,
    el preprocesador, las métricas de entrenamiento y las figuras de evaluación.
    
    Args:
        data_path (str): Ruta al archivo CSV que contiene los datos.
        model_output_path (str): Ruta donde se guardará el modelo entrenado.
        preprocessor_output_path (str): Ruta donde se guardará el preprocesador.
        metrics_output_path (str): Ruta donde se guardarán las métricas de entrenamiento en formato JSON.
        figures_output_path (str): Ruta donde se guardarán las figuras de evaluación.
    """
    #configuración de las variables de entorno para MLFlow
    #recordar qu debe existir un archivo .env en la raíz del proyecto con las variables de entorno necesarias
    #las credenciales de MLFlow se pueden obtener desde la interfaz web de MLFlow en la sección "settings" o "Configuration"
    load_dotenv(find_dotenv()) 
    mlflow.set_experiment("Proyecto 13MBID - v2 - Entrenamiento de modelos de clasificación")

    # Cargar los datos
    X_train, X_val, X_test, y_train, y_val, y_test, features_X, labels_y = load_data(data_path)

    # Crear el preprocesador
    preprocessor = create_preprocessor(features_X)

    
    #Definición del modelo a entrenar
    model = LogisticRegression(max_iter =1000)

    #Creación del pipeline que incluye el preprocesador y el modelo
    pipeline = Pipeline(steps=[('prep',preprocessor),('model',model)])
    
    #entrenamiento del modelo
    pipeline.fit(X_train,y_train)
    
    # Evaluar el modelo en el conjunto de test
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)
        
    #Calculo de metricas de evaluación
    metrics = {
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_precision_macro": precision_score(y_test, y_test_pred, average='macro',zero_division=0),
        "test_recall_macro": recall_score(y_test, y_test_pred, average='macro',zero_division=0),
        "test_ft_macro": f1_score(y_test, y_test_pred, average='macro',zero_division=0),
        "test_recall_negative": recall_score(y_test, y_test_pred, pos_label='negativo',average='macro', zero_division=0),
        "test_roc_aux_ovr": roc_auc_score(y_test, y_test_proba, multi_class='ovr',labels=pipeline.classes_),
    }

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}:{metric_value:.4f}")

    #Guardado de la matriz de confusión
    Path(figures_output_path).mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_test_pred, labels=pipeline.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusión Matrix - Logistic Regression - Test Set")
    cm_fig_path = f"{figures_output_path}/confusion_matrix_test.png"
    plt.savefig(cm_fig_path)
    plt.close()

    #registro general en MLFlow
    signature = infer_signature(X_train, pipeline.predict(X_train))

    with mlflow.start_run(run_name="Pipeline (prod) - Logistic Regression" ) as run:
        mlflow.log_params(model.get_params())
        mlflow.log_params({
            "train_sample_size": len(X_train),
            "validation_sample_size": len(X_val),
            "test_sample_size": len(X_test),
            "tfidf_max_features": 300 
        })

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_fig_path, artifact_path="figures")
        model_info = mlflow.sklearn.log_model(
            pipeline, name="model", signature=signature, input_example=X_train.head(3),
            registered_model_name=register_model_name,
        )

        run_id = mlflow.active_run().info.run_id
        print(f"Modelo registrado en MLflow a raíz del run ID: {run_id}")

    if register_model_name:
        version = getattr(model_info.registered_model_version, "version", model_info.registered_model_version)
        if metrics["test_accuracy"] >= UMBRAL_ACCURACY:
            mlflow.MlflowClient().set_registered_model_alias(
                name=register_model_name, version=version, alias=ALIAS_PRODUCCION
            )

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(preprocessor_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)


    #Guardado del modelo y entrenado del preprocesador
    joblib.dump(pipeline, model_output_path)
    joblib.dump(pipeline.named_steps["prep"], preprocessor_output_path)

    #Guardado de las metricas en un archivo JSON
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return pipeline, pipeline.named_steps['prep'], metrics

if __name__ == "__main__":
    train_model()

########################################################
# TODO: seleccionar el mejor modelo entre los candidatos basados en otras metricas
# en logar de solo accuracy y registrar el mejor modelo MLFlow 
# utilizando este script. Justificar la elección de las métricas y el modelo
# TODO: incorporar la imputación de valores faltantes dentro del pipeline,
# para que el modelo pueda recibir los datos sin procesarlos previamente
# en la instancia de producción 
# EXTRA: utilizar alguba técnica diferente, revisar hiperparametrización,
# considerar alternativas para el desbalanceo de las clases
# ######################################################   

    