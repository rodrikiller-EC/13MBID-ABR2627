# Importación de librerías y supresión de advertencias
from anyio import Path
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=UserWarning)

def visualize_data(datos_clientes: str = "data/raw/datos_clientes.csv",
                   datos_resenas: str = "data/raw/datos_resenas.csv",
                   output_dir: str = "docs/figures") -> None:
    """
    Función para visualizar los datos de clientes y reseñas.
    """
    #Crear directorio de salida si no existe
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    #lectura de los datos
    df_clientes = pd.read_csv(datos_clientes, sep=";")
    df_resenas = pd.read_csv(datos_resenas, sep=";")

    # Configuración general
    sns.set(style="whitegrid", palette="muted")
    

    #grafico 1: Distribución de edades de la variable target
    orden_valores_target = ["Negativo","Neutral","Positivo"]
    plt.rcParams["figure.figsize"] = (8, 5)
    sns.countplot(x="sentimiento", data=df_resenas, order=["Negativo", "Neutral", "Positivo"])
    plt.title("Distribución de la variable objetivo (sentimiento)")
    plt.xlabel("Sentimiento expresado en la reseña")
    plt.ylabel("Cantidad de reseñas")
    plt.savefig(output_dir / "distribucion_sentimiento.png")
    plt.close()

    #grafico 2: Lomgitud del texto de las reseñas
    df_resenas["longitud_texto"] = df_resenas["texto_resena"].str.len()

    sns.boxplot(x="sentimiento", y="longitud_texto", data=df_resenas, order=["Negativo", "Neutral", "Positivo"])
    plt.title("Longitud del texto de la reseña según sentimiento")
    plt.xlabel("Sentimiento")
    plt.ylabel("Cantidad de caracteres")
    plt.savefig(output_dir / "longitud_texto.png")
    plt.close()

    # se descarta la columna auxiliar, ya que solo se utilizó para esta visualización
    df_resenas.drop(columns=["longitud_texto"], inplace=True)

    #grafico 3: Matriz de correlación entre variables numéricas
    num_df = df_clientes.select_dtypes(include=['float64', 'int64'])
    corr = num_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de correlaciones - Clientes')
    plt.savefig(output_dir / "matriz_correlacion_clientes.png")
    plt.close()

    num_df = df_resenas.select_dtypes(include=['float64', 'int64'])
    corr = num_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de correlaciones - Reseñas')
    plt.savefig(output_dir / "matriz_correlacion_resenas.png")
    plt.close()

##############################################################################################
# TODO: Agregar almenos 2 visualizaciones más del escenario
#       Posiblemente alguna que vincule a alguno de los atributos con la variable objetivo.
#       Por ejemplo: sentimiento según tiempo de respuesta. 
#       Tambien puede ser la distribución de valores de otras variables
#------
#EXTRA (opcional): generar un reporte automático con una libreria como sweetviz, dtale
##############################################################################################

if __name__ == "__main__":
    visualize_data()

    
