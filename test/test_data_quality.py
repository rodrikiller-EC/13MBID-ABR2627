import pandas as pd
import pytest
from pandera.pandas import DataFrameSchema, Column, Check

GENEROS = ["Masculino", "Femenino", "Otro"]
SENTIMIENTOS = ["Positivo", "Negativo", "Neutral"]

@pytest.fixture
def datos_clientes():
    """Función para cargar los datos de clientes a un dataframe
    
    Args:
        None
    
    Returns: 
        pd.DataFrame: Dataframe con los datos de clientes
    """
    df = pd.read_csv("data/raw/datos_clientes.csv", sep=";")
    return df


@pytest.fixture
def datos_resenas():
    """Función para cargar los datos de reseñas a un dataframe
    
    Args:
        None
    
    Returns: 
        pd.DataFrame: Dataframe con los datos de reseñas
    """
    df = pd.read_csv("data/raw/datos_resenas.csv", sep=";")
    return df


def test_schema_datos_clientes(datos_clientes):
    """Función para validar el esquema de los datos de clientes
    
    Args:
        datos_clientes (pd.DataFrame): Dataframe con los datos de clientes
    
    Returns: 
        None
    """
    schema = DataFrameSchema(
        {
            "id_cliente": Column(int, nullable=False),
            "edad": Column(float, nullable=True, checks=Check.in_range(18,120)),
            "genero": Column(str, nullable=False, checks=Check.isin(GENEROS)),
            "ciudad": Column(str, nullable=False),
            "antiguedad_cliente_meses": Column(int, nullable=True, checks=Check.ge(0)),
            "canal_registro": Column(str, nullable=False)         
        }
    )
    #Lazy=True permite que se validen todas las columnas y se muestren 
    # todos los errores encontrados en lugar de detenerse en el primer error.

    schema.validate(datos_clientes, lazy=True)


def test_schema_datos_resenas(datos_resenas):
    """Función para validar el esquema de los datos de reseñas
    
    Args:
        datos_resenas (pd.DataFrame): Dataframe con los datos de reseñas
    
    Returns: 
        None
    """
    schema = DataFrameSchema(
        {
            "id_resena": Column(int, nullable=False),
            "id_cliente": Column(int, nullable=False),
            "fecha_resena": Column(str, nullable=False),
            "calificacion": Column(int, nullable=False, checks=Check.in_range(1,5)),
            "sentimiento": Column(str, nullable=True, checks=Check.isin(SENTIMIENTOS)),            
            # PENDIENTE: TODO agregar la validación de los atributos restantes
        }
    )
    #Lazy=True permite que se validen todas las columnas y se muestren 
    # todos los errores encontrados en lugar de detenerse en el primer error.
    schema.validate(datos_resenas, lazy=True)

def test_basicos_clientes(datos_clientes):
    """Función para validar los datos básicos de los clientes
    
    Args:
        datos_clientes (pd.DataFrame): Dataframe con los datos de clientes
    
    Returns: 
        None
    """
    # Validar que no haya valores nulos en las columnas obligatorias
    assert not datos_clientes.empty, "El dataframe de clientes está vacío"
    assert datos_clientes.shape[1] == 6, "El dataframe de clientes no tiene 6 columnas"
    assert datos_clientes["id_cliente"].notnull().all(), "la columna id_cliente tiene valores nulos"
    #assert datos_clientes["ciudad"].notnull().all(), "la columna ciudad tiene valores nulos"
    #assert datos_clientes["canal_registro"].notnull().all(), "la columna canal_registro tiene valores nulos"
    # EXTRA: pueden completar otras validaciones básicas


def test_basicos_resenas(datos_resenas):
    """Función para validar los datos básicos de las reseñas
    
    Args:
        datos_resenas (pd.DataFrame): Dataframe con los datos de reseñas
    
    Returns: 
        None
    """
    # Validar que no haya valores nulos en las columnas obligatorias
    assert not datos_resenas.empty, "El dataframe de reseñas está vacío"
    assert datos_resenas.shape[1] == 13, "El dataframe de reseñas no tiene 13 columnas"
    assert datos_resenas["id_resena"].notnull().all(), "la columna id_resena tiene valores nulos"
    assert datos_resenas["id_cliente"].notnull().all(), "la columna id_cliente tiene valores nulos"
    #assert datos_resenas["fecha_resena"].notnull().all(), "la columna fecha_resena tiene valores nulos"
    #assert datos_resenas["calificacion"].notnull().all(), "la columna calificacion tiene valores nulos"    
    # EXTRA: pueden completar otras validaciones básicas

def test_integridad_referencial(datos_clientes, datos_resenas):
    """Función para validar la integridad referencial entre los datos de clientes y reseñas
    
    Args:
        datos_clientes (pd.DataFrame): Dataframe con los datos de clientes
        datos_resenas (pd.DataFrame): Dataframe con los datos de reseñas
    
    Returns: 
        None
    """
    # Validar que todos los id_cliente en datos_resenas existan en datos_clientes
    df_ids = datos_resenas[['id_cliente']].merge(
        datos_clientes[['id_cliente']],
        on='id_cliente', 
        how='left', 
        indicator=True # genera la columna _merge con el resultado de la unión
    )
    integridad_referencial = DataFrameSchema( 
        
        {
            "_merge": Column(str, nullable=False, checks=Check.isin(["both"])),
        }
    )
    integridad_referencial.validate(df_ids, lazy=True)

    ###################################################################################
    #TODO: agregar almenos una función de test con una o mas validaciones más
    # allá de las estructuras básicas de cada dataset, por ejemplo: unicidad de
    # claves (id) y completar las evaluaciones de estructura pendientes (reseñas)
    # EXTRA: validar alguna regla de negocio como podría ser la consistencia entre
    # 'calificación' y 'sentimiento' entendiendo que una calificación baja se relaciona 
    # con un sentimiento negativo y lo inverso para la calificación alta.
    # Por ejemplo, si la calificación es menor a 3, el sentimiento no debería ser 
    # positivo en una proporción mayor al 5%.
    # ###################################################################################

