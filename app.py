import streamlit as st
import base64
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry
import requests, pytz
from streamlit_autorefresh import st_autorefresh
from neuralforecast.core import NeuralForecast

# Configuración inicial
def set_page_config():
    st.set_page_config(page_title="Monitoreo Energético IA", layout="wide")
    st.title("📈 Proyectos IA + Eficiencia energética")

def get_climate_data_act(lat, lon):
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
      "latitude": lat, "longitude": lon,
      "current": ["temperature_2m", "relative_humidity_2m", "precipitation"]
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    current = response.Current()
    current_temperature_2m = current.Variables(0).Value()
    current_relative_humidity_2m = current.Variables(1).Value()
    current_rain = current.Variables(2).Value()
    return {
        "temperatura": round(current_temperature_2m,2),
        "humedad": current_relative_humidity_2m,
        "lluvia_mm": current_rain
    }

def get_IA_model():
    IA_model = NeuralForecast.load(path='checkpoints/test_run')
    return IA_model

def get_climate_data_1m(lat, lon):
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
      "latitude": lat, "longitude": lon,
      "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation"], "past_days": 31
    }
    responses = openmeteo.weather_api(url, params=params)
    hourly = responses[0].Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_rain = hourly.Variables(2).ValuesAsNumpy()

    hourly_data = {"Fecha": pd.date_range(
      start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
      end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
      freq = pd.Timedelta(seconds = hourly.Interval()),
      inclusive = "left"
    )}

    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["precipitation"] = hourly_rain

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    return hourly_dataframe


def generar_datos(nombre, dias=30):
    fechas = pd.date_range(end=pd.Timestamp.today(), periods=dias)
    np.random.seed(hash(nombre) % 123456)  # Semilla distinta por medidor
    consumo = np.random.normal(loc=30, scale=10, size=dias).clip(min=0)
    return pd.DataFrame({"fecha": fechas, "consumo": consumo})

def graficar_consumo(df, titulo):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["fecha"], y=df["consumo"], mode="lines+markers"))
    fig.update_layout(title=titulo, xaxis_title="Fecha", yaxis_title="Consumo (kWh)", height=300)
    st.plotly_chart(fig, use_container_width=True)

def graficar_cond(df):
    x, y1, y2, y3 = df.columns
    df[x] = pd.to_datetime(df[x])
    now = pd.to_datetime(datetime.now())
    fig = go.Figure([
        go.Scatter(x=df[x], y=df[y1], mode="lines", name="Temperatura (°C)",  yaxis="y1", line=dict(color="#d62728")),
        go.Scatter(x=df[x], y=df[y2], mode="lines", name="Humedad Relativa (%)",  yaxis="y2", line=dict(color="#1f77b4")),
        go.Scatter(x=df[x], y=df[y3], mode="lines", name="Precipitaciones (mm)",  yaxis="y3", line=dict(color="#17becf"))
    ])
    fig.add_vline(x=now,line_width=2,line_dash="dash",line_color="black")
    fig.update_layout(title="", xaxis=dict(domain=[0.1, 0.99],title=x, showline=True, linecolor='black', showgrid=False, zeroline=False),
                      yaxis=dict(title="°C", side="left", autoshift=True, position = 0.09, showgrid=False, showline=False),
                      yaxis2=dict(title="%", overlaying="y", side="left", anchor="free", showgrid=False, showline=False, autoshift=True, position = 0.03),
                      yaxis3=dict(title="mm", overlaying="y", side="left", anchor="free", showgrid=False, showline=False, autoshift=True),
                      margin=dict(l=100, r=20, t=40, b=60),
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.4, yanchor="top"), height=400)
    st.plotly_chart(fig, use_container_width=True)

# Función para obtener los íconos
def get_icons():
    return {
        "general": "images/MedidorGen.png",
        "ac": "images/MedidorAA.png",
        "ssfv": "images/MedidorPV.png",
        "otros": "images/MedidorOtros.png"
    }

# Función para obtener las métricas
def get_metrics(general, ac, otros, ssfv):
    return {
        "general": {"potencia": 25.4, "energia": general.round(2)},
        "ac": {"potencia": 10.2, "energia": f"{ac:.2f}"},
        "ssfv": {"potencia": -4.1, "energia": f"{-ssfv:.2f}"},
        "otros": {"potencia": 19.3, "energia": f"{otros:.2f}"},
    }

# Submedidores dinámicos
def get_submedidores(metrics):
    return [k for k in metrics if k != "general"]

# Estilo CSS para Streamlit
def inject_css():
    css = '''
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');


    body, h1, h2, h3, h4, h5, h6, p, .stDataFrame, .stButton>button, .stMetricValue {
        font-family: 'Poppins' !important;
    }

    [data-testid="stMetric"] {
        width: fit-content!important;
        margin: auto;
    }

    [data-testid="stMetric"] > div {
        width: fit-content!important;
        margin: auto;
    }

    [data-testid="stMetric"] label {
        width: fit-content!important;
        margin: auto;
    }

    div[data-testid="stMetric"] > div:nth-child(2) {
        font-family: 'Poppins';
    }


    .stButton {
        display: flex;
        justify-content: center;
        margin-top: 2em;
    }

     .stButton > button {
        background-color: #fcbf06;
        border: none;
        padding: 0.6em 1.2em;
        border-radius: 8px;
        font-weight: bold;
        transition: background-color 0.3s ease;
        width: 80%;
        margin: auto;
    }

    .stButton > button:hover {
        background-color: #lightgold;
        color: white;
    }

    label[data-testid="stMetricLabel"] p{
        font-size: 1.3em !important;
        font-family: 'Poppins';
    }

    div[data-testid="stMetric"] {
        padding: 15px;
        color: white;
        font-family: 'Poppins';
    }

    button[data-baseweb="tab"] p{
        font-size:18px!important;
        font-weight: bold;
        padding:20px;
    }

     .st-key-styled_tabs button[aria-selected="true"]{
        background-color:#eeeeee;
        border-radius:5px;
     }

    .st-key-styled_tabs div[data-baseweb="tab-panel"]{
        border-radius:5px;
        padding:10px;
        background-color:#eeeeee;
    }
    '''
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

def mostrar_imagen(path, width=150):
    img64 = base64.b64encode(open(path, "rb").read()).decode()
    st.markdown(
        f'<div style="text-align:center;"><img src="data:image/png;base64,{img64}" width="{width}"></div>',
        unsafe_allow_html=True
    )

def toggle_visibility(key):
    if key not in st.session_state:
        st.session_state[key] = True
    else:
        st.session_state[key] = not st.session_state[key]

# Función para mostrar el medidor general
def display_general(icons, metrics):
    colg = st.columns([1, 2, 1])[1]
    with colg:
        with st.container(border=True,):
            ca,cb = st.columns([1, 1], vertical_alignment='bottom')
            with ca:
                mostrar_imagen(icons['general'], 100)
            with cb:
                st.metric(label="Energía (kWh)", value=metrics['general']['energia'])

            if st.button("Ver Detalle", key="butt_gen",use_container_width=True):
                toggle_visibility("vis_gen")
            if st.session_state.get("vis_gen", False):
                df = generar_datos("general")
                graficar_consumo(df, "Consumo del Medidor General")


# Función para mostrar los submedidores
def display_submedidores(submedidores, nombres_submedidores, icons, metrics):
    cols = st.columns(len(submedidores))
    for i, label in enumerate(submedidores):
        with cols[i]:
            with st.container(border=True):
                ca,cb = st.columns([1, 1], vertical_alignment='bottom')
                with ca:
                    mostrar_imagen(icons[label], 100)
                with cb:
                    st.metric(label="Energía (kWh)", value=metrics[label]['energia'])
                st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)

                nombre = nombres_submedidores.get(label, label)
                key_btn = f"butt_{nombre}"
                key_vis = f"vis_{nombre}"

                if st.button("Ver Detalle", key=key_btn,use_container_width=True):
                    toggle_visibility(key_vis)

                if st.session_state.get(key_vis, False):
                    df = generar_datos(nombre)
                    graficar_consumo(df, f"Consumo de {nombre}")


def mostrar_interfaz_clima(datos, con_boton=False, lat=None, lon=None):
    with st.container(border=True):
        st.subheader("Condiciones Externas")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌡️ Temperatura", f"{round(datos['temperatura'],2)} °C")
        with col2:
            st.metric("💧 Humedad Relativa", f"{round(datos['humedad'],2)} %")
        with col3:
            st.metric("🌧️ Precipitaciones", f"{round(datos['lluvia_mm'],2)} mm")
        if con_boton:
            if st.button("Ver Detalle", key="butt_clima", use_container_width=True):
                toggle_visibility("vis_clima")

            if st.session_state.get("vis_clima", False):
                graficar_cond(get_climate_data_1m(lat, lon))

# Método principal
def main():
    set_page_config()
    st_autorefresh(interval=180000, key="clima_autorefresh")
    icons = get_icons()

    lat, lon = 3.4793789, -76.5287221
    datos = get_climate_data_act(lat, lon)
    with st.container(key="styled_tabs"):
        tab1, tab2 = st.tabs(["⚡ Medición Inteligente No Intrusiva ", " Smart Building "])
    nombres_submedidores = {
        "ac": "Aires Acondicionados",
        "ssfv": "SSFV",
        "otros": "Otras Cargas"
    }

    futr_df = pd.DataFrame({
    'ds':   [pd.to_datetime('2025-04-07 13:45'), pd.to_datetime('2025-04-07 13:45'), pd.to_datetime('2025-04-07 13:45')],  # siguiente timestamp
    'unique_id': ['Energia_kWh_AA','Energia_kWh_Otros','Energia_kWh_PV'],
    'Energia_kWh_General': [4.672,  4.672,  4.672],
    'Hour':  [13, 13, 13],
    'DOW':  [1, 1, 1],
    'RH2M': [datos["humedad"],datos["humedad"],datos["humedad"]],
    'T2M': [datos["temperatura"], datos["temperatura"], datos["temperatura"]]
    })

    #st.write(futr_df)
    modelo_IA = get_IA_model()
    Y_hat_df2 = modelo_IA.predict(futr_df=futr_df)
    metrics = get_metrics(futr_df.Energia_kWh_General[0],
                          Y_hat_df2.AutoLSTM[0],
                          Y_hat_df2.AutoLSTM[1],
                          Y_hat_df2.AutoLSTM[2])

    inject_css()

    with tab1.container(key='cont-nilm'):
        mostrar_interfaz_clima(datos)
        submedidores = get_submedidores(metrics)
        with st.container(border=True):
            st.subheader("Medición Desagregada")
            display_general(icons, metrics)
            st.markdown("<div style='text-align:center; margin-top: -20px; font-size:33px;'> ┌──────────────┼──────────────┐<br></div>", unsafe_allow_html=True)
            display_submedidores(submedidores, nombres_submedidores, icons, metrics)

    with tab2.container(key='cont-ci'):
        mostrar_interfaz_clima(datos,True,lat,lon)

    zona = pytz.timezone("America/Bogota")
    ahora = datetime.now(zona).strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"🔄 Esta página se actualiza cada 15 minutos. Última actualización: {ahora}")

if __name__ == "__main__":
    main()
