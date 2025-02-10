import streamlit as st  # ⬅️ Importar primero Streamlit
st.set_page_config(page_title="Análisis de Clientes", layout="wide")  # ⬅️ Debe ir justo después

# -*- coding: utf-8 -*-  

import pandas as pd
import numpy as np
import geopandas as gpd
import geodatasets as gds
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

"""datafrem"""

ruta  = "https://raw.githubusercontent.com/gabrielawad/programacion-para-ingenieria/refs/heads/main/archivos-datos/aplicaciones/analisis_clientes.csv"
clientes = pd.read_csv(ruta )
print(clientes)

"""**Interpolar**"""

def llenar_valores_faltantes(df):
    """
    Llena valores NaN en el DataFrame según relaciones establecidas.
    """
    # Obtener el nombre más común en toda la columna
    nombre_mas_comun = df["Nombre"].mode().iloc[0]

    # Llenar los valores NaN en "Nombre" con el nombre más común
    df["Nombre"] = df["Nombre"].fillna(nombre_mas_comun)

    # Calcular la moda de cada grupo de 'Nombre' para llenar 'Género'
    modas_genero = df.groupby("Nombre")["Género"].agg(pd.Series.mode)
    modas_genero = modas_genero.explode().drop_duplicates()
    df["Género"] = df["Género"].fillna(df["Nombre"].map(modas_genero))

    # Llenado global en caso de valores faltantes
    df["Género"] = df["Género"].fillna(df["Género"].mode().iloc[0])

    # Llenar Ingreso_Anual_USD según el promedio del Género o la media global
    df["Ingreso_Anual_USD"] = df["Ingreso_Anual_USD"].fillna(
        df.groupby("Género")["Ingreso_Anual_USD"].transform("mean")
    ).fillna(df["Ingreso_Anual_USD"].mean())

    # Llenar Edad según el promedio de personas con ingresos similares
    df["Edad"] = df["Edad"].fillna(
        df.groupby("Ingreso_Anual_USD")["Edad"].transform("mean")
    ).fillna(df["Edad"].mean())

    # Llenar Frecuencia_Compra con la moda de personas con historial de compras similar
    modas_frecuencia = df.groupby("Historial_Compras")["Frecuencia_Compra"].agg(pd.Series.mode)
    modas_frecuencia = modas_frecuencia.explode().drop_duplicates()
    df["Frecuencia_Compra"] = df["Frecuencia_Compra"].fillna(df["Historial_Compras"].map(modas_frecuencia))

    # Llenado global en caso de valores faltantes
    df["Frecuencia_Compra"] = df["Frecuencia_Compra"].fillna(df["Frecuencia_Compra"].mode().iloc[0])

    # Llenar Historial_Compras con la media de personas con frecuencia de compra similar
    df["Historial_Compras"] = df["Historial_Compras"].fillna(
        df.groupby("Frecuencia_Compra")["Historial_Compras"].transform("mean")
    ).fillna(df["Historial_Compras"].mean())

    # Llenar Latitud y Longitud según la media por Género o media global
    df["Latitud"] = df["Latitud"].fillna(
        df.groupby("Género")["Latitud"].transform("mean")
    ).fillna(df["Latitud"].mean())

    df["Longitud"] = df["Longitud"].fillna(
        df.groupby("Género")["Longitud"].transform("mean")
    ).fillna(df["Longitud"].mean())

    return df

def contar_nans(df):
    """
    Devuelve la cantidad total de valores NaN en el DataFrame.
    """
    return df.isna().sum().sum()

# Aplicar la función para llenar valores NaN
clientes2 = clientes.copy()  # Evitar modificaciones en el original
clientes2 = llenar_valores_faltantes(clientes2)
clientes2 = llenar_valores_faltantes(clientes2)

# Contar valores NaN restantes
print("Valores NaN restantes:", contar_nans(clientes2))
print(clientes2)
print(clientes2.isna().sum())

def cargar():
  clientes = pd.read_csv(ruta )
  clientes2 = clientes.copy()  # Evitar modificaciones en el original
  clientes2 = llenar_valores_faltantes(clientes2)
  clientes2 = llenar_valores_faltantes(clientes2)
  return clientes2

"""funciones"""

def graficar_suramerica(url):
    """Carga un shapefile desde una URL y devuelve un GeoDataFrame
    con los países de Suramérica.

    Args:
        url (str): URL del archivo shapefile en formato ZIP.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame con los países de Suramérica.
    """
    world = gpd.read_file(f"/vsizip/vsicurl/{url}")


    paises_suramerica = [
        # Sudamérica
        "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
        "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela",
        # Centroamérica
        "Belize", "Costa Rica", "El Salvador", "Guatemala", "Honduras",
        "Mexico", "Nicaragua", "Panama"
    ]

    return world[world["ADMIN"].isin(paises_suramerica)]

# URL del shapefile (ejemplo)
url = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"

# Obtener el GeoDataFrame de Suramérica



def correlacion_edad_ingreso(df):
    """Calcula la correlación entre edad e ingreso anual globalmente y por grupos."""
    correlaciones = {
        "global": df["Edad"].corr(df["Ingreso_Anual_USD"]),
        "por_genero": df.groupby("Género")[["Edad", "Ingreso_Anual_USD"]].corr().unstack().iloc[:, 1],
        "por_frecuencia": df.groupby("Frecuencia_Compra")[["Edad", "Ingreso_Anual_USD"]].corr().unstack().iloc[:, 1],
    }
    return correlaciones



def mapa_ubicacion(df, url_shapefile, filtro_col=None, filtro_valor=None):
    """
    Genera un mapa de ubicación de clientes con opción de filtrado.

    Args:
        df (pd.DataFrame): DataFrame con columnas 'Latitud' y 'Longitud'.
        url_shapefile (str): URL del archivo shapefile en formato ZIP.
        filtro_col (str, opcional): Columna para filtrar datos (ej. 'Género').
        filtro_valor (str, opcional): Valor a filtrar (ej. 'Masculino').

    Returns:
        None
    """
    # Cargar el mapa de Sudamérica y asegurarse de que está en EPSG:4326
    suramerica = graficar_suramerica(url_shapefile)

    # Crear GeoDataFrame de clientes en EPSG:4326
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitud"], df["Latitud"]))
    gdf.set_crs("EPSG:4326", inplace=True)  # Asegurar la proyección correcta

    # Aplicar filtro si se especifica
    if filtro_col and filtro_col in df.columns:
        gdf = gdf[gdf[filtro_col] == filtro_valor]

    # Graficar
    fig, ax = plt.subplots(figsize=(10, 10))
    suramerica.plot(ax=ax, edgecolor="black", color="lightgray")  # Mapa base
    gdf.plot(ax=ax, markersize=5, color="red", alpha=0.7, edgecolor="black")  # Clientes

    plt.title("Mapa de Ubicación de Clientes")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.show()


def mapa_personalizado(df, filtros,url):
    """Genera un mapa con filtros personalizados en hasta 4 variables."""
    df_filtrado = df.copy()
    for var, (min_val, max_val) in filtros.items():
        df_filtrado = df_filtrado[df_filtrado[var].between(min_val, max_val)]
    mapa_ubicacion(df_filtrado,url)


def cluster_frecuencia_compra(df):
    """Agrupa clientes por frecuencia de compra y visualiza en un gráfico de barras."""
    conteo = df["Frecuencia_Compra"].value_counts()
    conteo.plot(kind="bar", figsize=(8, 5))
    plt.title("Clusters por Frecuencia de Compra")
    plt.xlabel("Frecuencia de Compra")
    plt.ylabel("Cantidad de Clientes")
    plt.show()


def grafico_barras_genero_frecuencia(df):
    """Genera un gráfico de barras agrupado por género y frecuencia de compra."""
    df.groupby(["Género", "Frecuencia_Compra"]).size().unstack().plot(kind="bar", figsize=(10, 6))
    plt.title("Distribución de Clientes por Género y Frecuencia de Compra")
    plt.ylabel("Cantidad de Clientes")
    plt.show()


def mapa_calor_ingresos(df):
    """Genera un mapa de calor basado en ingresos anuales."""
    plt.figure(figsize=(8, 6))
    plt.hexbin(df["Longitud"], df["Latitud"], C=df["Ingreso_Anual_USD"], gridsize=50, cmap="coolwarm", reduce_C_function=np.mean)
    plt.colorbar(label="Ingreso Anual Promedio")
    plt.title("Mapa de Calor de Ingresos")
    plt.show()


def calcular_distancias(df, top_n=10):
    """Calcula distancias entre los compradores de mayores ingresos."""
    top_compradores = df.nlargest(top_n, "Ingreso_Anual_USD")[["Latitud", "Longitud"]]
    dist_matrix = squareform(pdist(top_compradores))
    return pd.DataFrame(dist_matrix, index=top_compradores.index, columns=top_compradores.index)

"""prueba"""

corelaciomnes = correlacion_edad_ingreso(clientes2)
print("Correlaciones:", corelaciomnes)
mapa_ubicacion(clientes2,url)
mapa_personalizado(clientes2, {"Edad": (20, 40), "Ingreso_Anual_USD": (0, 100000)},url)
cluster_frecuencia_compra(clientes2)
grafico_barras_genero_frecuencia(clientes2)
mapa_calor_ingresos(clientes2)
distancias = calcular_distancias(clientes2)

"""Aplicacion"""

def main():
    st.title("📊 Análisis de Clientes y Finanzas")

    df = cargar()

    menu = [
        "Análisis de Correlación", "Mapa de Ubicación", "Mapa Personalizado",
        "Clúster de Frecuencia de Compra", "Gráfico de Barras",
        "Mapa de Calor de Ingresos", "Distancias entre Compradores"
    ]

    opcion = st.sidebar.selectbox("Selecciona una opción", menu)

    if opcion == "Análisis de Correlación":
        st.subheader("📈 Correlación entre Edad e Ingreso Anual")
        correlaciones = correlacion_edad_ingreso(df)
        st.write("Correlación Global:", correlaciones["global"])
        st.write("Correlación por Género:", correlaciones["por_genero"])
        st.write("Correlación por Frecuencia de Compra:", correlaciones["por_frecuencia"])

    elif opcion == "Mapa de Ubicación":
        st.subheader("🌍 Mapa de Ubicación de Clientes")
        filtro_col = st.selectbox("Filtrar por", [None, "Género", "Frecuencia_Compra"])
        filtro_valor = None
        if filtro_col:
            filtro_valor = st.selectbox("Selecciona el valor", df[filtro_col].unique())
        mapa_ubicacion(df, filtro_col, filtro_valor)

    elif opcion == "Mapa Personalizado":
        st.subheader("📌 Mapa Personalizado")
        filtros = {}
        for col in ["Latitud", "Longitud", "Ingreso_Anual_USD", "Edad"]:
            min_val, max_val = float(df[col].min()), float(df[col].max())
            rango = st.slider(f"Rango para {col}", min_val, max_val, (min_val, max_val))
            filtros[col] = rango
        mapa_personalizado(df, filtros)

    elif opcion == "Clúster de Frecuencia de Compra":
        st.subheader("🛒 Clúster de Frecuencia de Compra")
        cluster_frecuencia_compra(df)

    elif opcion == "Gráfico de Barras":
        st.subheader("📊 Gráfico de Barras por Género y Frecuencia de Compra")
        grafico_barras_genero_frecuencia(df)

    elif opcion == "Mapa de Calor de Ingresos":
        st.subheader("🔥 Mapa de Calor de Ingresos")
        mapa_calor_ingresos(df)

    elif opcion == "Distancias entre Compradores":
        st.subheader("📏 Cálculo de Distancias entre los Compradores de Mayores Ingresos")
        top_n = st.slider("Cantidad de compradores a analizar", 5, 50, 10)
        distancias = calcular_distancias(df, top_n)
        st.write(distancias)

if __name__ == "__main__":
    main()
