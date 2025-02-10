import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

st.set_page_config(page_title="📊 Análisis de Clientes y Finanzas", layout="wide")

# Cargar datos
ruta = "https://raw.githubusercontent.com/gabrielawad/programacion-para-ingenieria/refs/heads/main/archivos-datos/aplicaciones/analisis_clientes.csv"

@st.cache_data
def cargar():
    df = pd.read_csv(ruta)
    return llenar_valores_faltantes(df)

def llenar_valores_faltantes(df):
    df["Nombre"] = df["Nombre"].fillna(df["Nombre"].mode().iloc[0])
    modas_genero = df.groupby("Nombre")["Género"].agg(pd.Series.mode).explode().drop_duplicates()
    df["Género"] = df["Género"].fillna(df["Nombre"].map(modas_genero))
    df["Género"] = df["Género"].fillna(df["Género"].mode().iloc[0])
    df["Ingreso_Anual_USD"] = df["Ingreso_Anual_USD"].fillna(df.groupby("Género")["Ingreso_Anual_USD"].transform("mean")).fillna(df["Ingreso_Anual_USD"].mean())
    df["Edad"] = df["Edad"].fillna(df.groupby("Ingreso_Anual_USD")["Edad"].transform("mean")).fillna(df["Edad"].mean())
    df["Latitud"] = df["Latitud"].fillna(df.groupby("Género")["Latitud"].transform("mean")).fillna(df["Latitud"].mean())
    df["Longitud"] = df["Longitud"].fillna(df.groupby("Género")["Longitud"].transform("mean")).fillna(df["Longitud"].mean())
    return df

def graficar_suramerica():
    url = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"
    world = gpd.read_file(f"/vsizip/vsicurl/{url}")
    paises = ["Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
              "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela",
              "Belize", "Costa Rica", "El Salvador", "Guatemala", "Honduras",
              "Mexico", "Nicaragua", "Panama"]
    return world[world["ADMIN"].isin(paises)]

def correlacion_edad_ingreso(df):
    return {
        "Global": df["Edad"].corr(df["Ingreso_Anual_USD"]),
        "Por Género": df.groupby("Género")[["Edad", "Ingreso_Anual_USD"]].corr().unstack().iloc[:, 1],
        "Por Frecuencia": df.groupby("Frecuencia_Compra")[["Edad", "Ingreso_Anual_USD"]].corr().unstack().iloc[:, 1],
    }

def mapa_ubicacion(df, filtro_col=None, filtro_valor=None):
    suramerica = graficar_suramerica()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitud"], df["Latitud"]))
    gdf.set_crs("EPSG:4326", inplace=True)

    if filtro_col:
        gdf = gdf[gdf[filtro_col] == filtro_valor]

    fig, ax = plt.subplots(figsize=(10, 10))
    suramerica.plot(ax=ax, edgecolor="black", color="lightgray")
    gdf.plot(ax=ax, markersize=5, color="red", alpha=0.7, edgecolor="black")
    plt.title("📍 Ubicación de Clientes")
    st.pyplot(fig)

def cluster_frecuencia_compra(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    df["Frecuencia_Compra"].value_counts().plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
    plt.title("🛒 Clúster de Frecuencia de Compra")
    plt.xlabel("Frecuencia de Compra")
    plt.ylabel("Cantidad de Clientes")
    st.pyplot(fig)

def grafico_barras_genero_frecuencia(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    df.groupby(["Género", "Frecuencia_Compra"]).size().unstack().plot(kind="bar", ax=ax, color=["#FFA07A", "#20B2AA"])
    plt.title("📊 Clientes por Género y Frecuencia de Compra")
    plt.ylabel("Cantidad de Clientes")
    st.pyplot(fig)

def mapa_calor_ingresos(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(df["Longitud"], df["Latitud"], C=df["Ingreso_Anual_USD"], gridsize=50, cmap="coolwarm", reduce_C_function=np.mean)
    plt.colorbar(hb, label="Ingreso Anual Promedio")
    plt.title("🔥 Mapa de Calor de Ingresos")
    st.pyplot(fig)

def calcular_distancias(df, top_n=10):
    top_compradores = df.nlargest(top_n, "Ingreso_Anual_USD")[["Latitud", "Longitud"]]
    dist_matrix = squareform(pdist(top_compradores))
    return pd.DataFrame(dist_matrix, index=top_compradores.index, columns=top_compradores.index)

def main():
    st.title("📊 Análisis de Clientes y Finanzas")
    df = cargar()

    menu = ["🏠 Inicio", "📈 Análisis de Correlación", "🌍 Mapa de Ubicación",
            "🛒 Clúster de Frecuencia", "📊 Gráfico de Barras", "🔥 Mapa de Calor de Ingresos",
            "📏 Distancias entre Compradores"]

    opcion = st.sidebar.radio("📌 Selecciona una opción", menu)

    if opcion == "🏠 Inicio":
        st.markdown("## 🏠 Bienvenido al Panel de Análisis")
        st.write("Este panel te permite visualizar datos sobre clientes, incluyendo análisis de correlación, ubicaciones y tendencias de compra.")

    elif opcion == "📈 Análisis de Correlación":
        st.markdown("## 📈 Correlación entre Edad e Ingreso Anual")
        corr = correlacion_edad_ingreso(df)
        col1, col2 = st.columns(2)
        with col1:
            st.write("### 📊 Correlación Global", corr["Global"])
        with col2:
            st.write("### 📊 Correlación por Género", corr["Por Género"])
        st.write("### 📊 Correlación por Frecuencia de Compra")
        st.dataframe(corr["Por Frecuencia"])

    elif opcion == "🌍 Mapa de Ubicación":
        st.markdown("## 🌍 Mapa de Ubicación de Clientes")
        filtro_col = st.selectbox("Filtrar por", [None, "Género", "Frecuencia_Compra"])
        if filtro_col:
            filtro_valor = st.selectbox("Selecciona el valor", df[filtro_col].unique())
        mapa_ubicacion(df, filtro_col, filtro_valor)

    elif opcion == "🛒 Clúster de Frecuencia":
        st.markdown("## 🛒 Clúster de Frecuencia de Compra")
        cluster_frecuencia_compra(df)

    elif opcion == "📊 Gráfico de Barras":
        st.markdown("## 📊 Distribución de Clientes")
        grafico_barras_genero_frecuencia(df)

    elif opcion == "🔥 Mapa de Calor de Ingresos":
        st.markdown("## 🔥 Mapa de Calor de Ingresos")
        mapa_calor_ingresos(df)

    elif opcion == "📏 Distancias entre Compradores":
        st.markdown("## 📏 Distancias entre los Compradores de Mayores Ingresos")
        st.dataframe(calcular_distancias(df))

if __name__ == "__main__":
    main()
