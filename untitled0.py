import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

st.set_page_config(page_title="Análisis de Clientes", layout="wide")

# Cargar datos
ruta = "https://raw.githubusercontent.com/gabrielawad/programacion-para-ingenieria/refs/heads/main/archivos-datos/aplicaciones/analisis_clientes.csv"

def cargar():
    """Carga los datos y llena valores faltantes."""
    df = pd.read_csv(ruta)
    return llenar_valores_faltantes(df)

def llenar_valores_faltantes(df):
    """Llena valores NaN en el DataFrame según relaciones establecidas."""
    df["Nombre"] = df["Nombre"].fillna(df["Nombre"].mode().iloc[0])
    modas_genero = df.groupby("Nombre")["Género"].agg(pd.Series.mode).explode().drop_duplicates()
    df["Género"] = df["Género"].fillna(df["Nombre"].map(modas_genero))
    df["Género"] = df["Género"].fillna(df["Género"].mode().iloc[0])
    df["Ingreso_Anual_USD"] = df["Ingreso_Anual_USD"].fillna(df.groupby("Género")["Ingreso_Anual_USD"].transform("mean")).fillna(df["Ingreso_Anual_USD"].mean())
    df["Edad"] = df["Edad"].fillna(df.groupby("Ingreso_Anual_USD")["Edad"].transform("mean")).fillna(df["Edad"].mean())
    modas_frecuencia = df.groupby("Historial_Compras")["Frecuencia_Compra"].agg(pd.Series.mode).explode().drop_duplicates()
    df["Frecuencia_Compra"] = df["Frecuencia_Compra"].fillna(df["Historial_Compras"].map(modas_frecuencia))
    df["Frecuencia_Compra"] = df["Frecuencia_Compra"].fillna(df["Frecuencia_Compra"].mode().iloc[0])
    df["Historial_Compras"] = df["Historial_Compras"].fillna(df.groupby("Frecuencia_Compra")["Historial_Compras"].transform("mean")).fillna(df["Historial_Compras"].mean())
    df["Latitud"] = df["Latitud"].fillna(df.groupby("Género")["Latitud"].transform("mean")).fillna(df["Latitud"].mean())
    df["Longitud"] = df["Longitud"].fillna(df.groupby("Género")["Longitud"].transform("mean")).fillna(df["Longitud"].mean())
    return df

def graficar_suramerica():
    """Carga un shapefile desde una URL y devuelve un GeoDataFrame de Suramérica."""
    url = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"
    world = gpd.read_file(f"/vsizip/vsicurl/{url}")
    paises = ["Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
              "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela",
              "Belize", "Costa Rica", "El Salvador", "Guatemala", "Honduras",
              "Mexico", "Nicaragua", "Panama"]
    return world[world["ADMIN"].isin(paises)]

def correlacion_edad_ingreso(df):
    """Calcula la correlación entre edad e ingreso anual globalmente y por grupos."""
    correlaciones = {
        "global": df["Edad"].corr(df["Ingreso_Anual_USD"]),
        "por_genero": df.groupby("Género")[["Edad", "Ingreso_Anual_USD"]].corr().unstack().iloc[:, 1],
        "por_frecuencia": df.groupby("Frecuencia_Compra")[["Edad", "Ingreso_Anual_USD"]].corr().unstack().iloc[:, 1],
    }
    return correlaciones

def mapa_ubicacion(df, filtro_col=None, filtro_valor=None):
    """Genera un mapa de ubicación de clientes con opción de filtrado."""
    suramerica = graficar_suramerica()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitud"], df["Latitud"]))
    gdf.set_crs("EPSG:4326", inplace=True)

    if filtro_col and filtro_col in df.columns:
        gdf = gdf[gdf[filtro_col] == filtro_valor]

    fig, ax = plt.subplots(figsize=(10, 10))
    suramerica.plot(ax=ax, edgecolor="black", color="lightgray")
    gdf.plot(ax=ax, markersize=5, color="red", alpha=0.7, edgecolor="black")

    plt.title("Mapa de Ubicación de Clientes")
    st.pyplot(fig)

def cluster_frecuencia_compra(df):
    """Agrupa clientes por frecuencia de compra y visualiza en un gráfico de barras."""
    fig, ax = plt.subplots(figsize=(8, 5))
    df["Frecuencia_Compra"].value_counts().plot(kind="bar", ax=ax)
    plt.title("Clusters por Frecuencia de Compra")
    plt.xlabel("Frecuencia de Compra")
    plt.ylabel("Cantidad de Clientes")
    st.pyplot(fig)

def grafico_barras_genero_frecuencia(df):
    """Genera un gráfico de barras agrupado por género y frecuencia de compra."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df.groupby(["Género", "Frecuencia_Compra"]).size().unstack().plot(kind="bar", ax=ax)
    plt.title("Distribución de Clientes por Género y Frecuencia de Compra")
    plt.ylabel("Cantidad de Clientes")
    st.pyplot(fig)

def mapa_calor_ingresos(df):
    """Genera un mapa de calor basado en ingresos anuales."""
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(df["Longitud"], df["Latitud"], C=df["Ingreso_Anual_USD"], gridsize=50, cmap="coolwarm", reduce_C_function=np.mean)
    plt.colorbar(hb, label="Ingreso Anual Promedio")
    plt.title("Mapa de Calor de Ingresos")
    st.pyplot(fig)

def calcular_distancias(df, top_n=10):
    """Calcula distancias entre los compradores de mayores ingresos."""
    top_compradores = df.nlargest(top_n, "Ingreso_Anual_USD")[["Latitud", "Longitud"]]
    dist_matrix = squareform(pdist(top_compradores))
    return pd.DataFrame(dist_matrix, index=top_compradores.index, columns=top_compradores.index)

def main():
    st.title("📊 Análisis de Clientes y Finanzas")
    df = cargar()

    menu = ["Análisis de Correlación", "Mapa de Ubicación", "Clúster de Frecuencia", 
            "Gráfico de Barras", "Mapa de Calor de Ingresos", "Distancias entre Compradores"]

    opcion = st.sidebar.selectbox("Selecciona una opción", menu)

    if opcion == "Análisis de Correlación":
        st.subheader("📈 Correlación entre Edad e Ingreso Anual")
        st.write(correlacion_edad_ingreso(df))

    elif opcion == "Mapa de Ubicación":
        st.subheader("🌍 Mapa de Ubicación de Clientes")
        filtro_col = st.selectbox("Filtrar por", [None, "Género", "Frecuencia_Compra"])
        filtro_valor = None
        if filtro_col:
            filtro_valor = st.selectbox("Selecciona el valor", df[filtro_col].unique())
        mapa_ubicacion(df, filtro_col, filtro_valor)

    elif opcion == "Clúster de Frecuencia":
        st.subheader("🛒 Clúster de Frecuencia de Compra")
        cluster_frecuencia_compra(df)

    elif opcion == "Gráfico de Barras":
        grafico_barras_genero_frecuencia(df)

    elif opcion == "Mapa de Calor de Ingresos":
        mapa_calor_ingresos(df)

    elif opcion == "Distancias entre Compradores":
        st.write(calcular_distancias(df))

if __name__ == "__main__":
    main()
