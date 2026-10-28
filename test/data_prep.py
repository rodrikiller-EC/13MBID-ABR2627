import pandas as pd
from pathlib import Path

def process_data(
        datos_clientes: str = "daa/raw/datos_clientes.csv",
        datos_resenas: str = "daa/raw/datos_resenas.csv",
        output_dir: str = "daa/processed",
    )-> None:
    """
    Función para procesar los datos de clientes y reseñas, y guardar los resultados en un directorio de salida.
    Args:
        datos_clientes (str): Ruta al archivo csv con los datos de clientes.
        datos_resenas (str): Ruta al archivo csv con los datos de reseñas.
        output_dir (str): El directorio donde se guardarán los datos procesados.

    Returns:
        None    
    """

    # Lectura de los datos
    df_clientes = pd.read_csv("../data/raw/datos_clientes.csv", sep=";")
    df_resenas = pd.read_csv("../data/raw/datos_resenas.csv", sep=";")

    #############################################################################
    # Selección de los datos (filtro de columnas)
    #############################################################################

    col_eliminar_clientes = ['canal_registro']
    df_clientes.drop(col_eliminar_clientes, inplace=True, axis=1)

    #############################################################################
    # Limpieza de los datos (filtro de filas)
    #############################################################################
    df_clientes_filtrado = df_clientes[(df_clientes['edad'] < 90) | (df_clientes['edad'].isna())]   
    df_resenas_filtrado = df_resenas.drop_duplicates().copy()

    #############################################################################
    # tratamiento de valores nulos
    ###############################################################################
    
    df_clientes_filtrado['edad'] = df_clientes_filtrado.groupby("ciudad")["edad"]\
                        .transform(lambda x: x.fillna(x.median()))
    df_resenas_filtrado['tiempo_respuesta_horas'] = df_resenas_filtrado.groupby("categoria_producto")["tiempo_respuesta_horas"]\
                        .transform(lambda x: x.fillna(x.median()))

    