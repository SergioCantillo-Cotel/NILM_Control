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
from keras.models import load_model
from google.oauth2 import service_account
from pandas_gbq import read_gbq

# Configuración inicial
credenciales_json = st.secrets["gcp_service_account"]
BIGQUERY_PROJECT_ID = st.secrets["bigquery"]["project_id"]
BIGQUERY_DATASET_ID = st.secrets["bigquery"]["dataset_id"]
TABLA = st.secrets["bigquery"]["table"]
TABLA_COMPLETA = f"{BIGQUERY_DATASET_ID}.{TABLA}"

def set_page_config():
    st.set_page_config(page_title="Monitoreo Energético IA", layout="wide")
    st.title("📈 Proyectos IA + Eficiencia energética")

def get_IA_model():
    #IA_model = load_model('models/NILM_Model_best.keras')
    IA_model = load_model('models/NILM_Model.keras')
    return IA_model

def bigquery_auth():
    return service_account.Credentials.from_service_account_info(credenciales_json)

def read_bq_db(credentials):
    query = f"SELECT * FROM `{TABLA_COMPLETA}` ORDER BY datetime_record ASC"
    df = read_gbq(query, project_id=BIGQUERY_PROJECT_ID, credentials=credentials).rename(columns={'id': 'unique_id', 'datetime_record': 'ds'})
    mapping = {'PM_General_Potencia_Activa_Total': 'General',
               'PM_Aires_Potencia_Activa_Total':  'Aires Acondicionados',
               'Inversor_Solar_Potencia_Salida':  'SSFV'}

    df = df[df['unique_id'].isin(mapping)]
    df['unique_id'] = df['unique_id'].map(mapping)
    df['ds'] = pd.to_datetime(df['ds']).dt.floor('15min')
    df['value'] *= 0.25
    df = gen_others_load(df)
    return df

def gen_others_load(df):
    pivot = (df.pivot_table(index=['ds', 'unit', 'company', 'headquarters'],columns='unique_id',values='value').reset_index())
    pivot['Otros'] = pivot['General'] + pivot['SSFV'] - pivot['Aires Acondicionados']
    result = (pivot.melt(id_vars=['ds', 'unit', 'company', 'headquarters'],value_vars=['General', 'Aires Acondicionados', 'SSFV', 'Otros'],var_name='unique_id',value_name='value').sort_values(['ds', 'unique_id']).reset_index(drop=True))
    return result


def get_climate_data_1m(lat, lon):
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
      "latitude": lat, "longitude": lon,
      "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation"],
      "start_date": "2025-05-15",
	    "end_date": ""+datetime.now().strftime("%Y-%m-%d"),
    }
    responses = openmeteo.weather_api(url, params=params)
    hourly = responses[0].Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_rain = hourly.Variables(2).ValuesAsNumpy()

    hourly_data = {"ds": pd.date_range(
      start = pd.to_datetime(hourly.Time(), unit = "s", utc = False),
      end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = False),
      freq = pd.Timedelta(seconds = hourly.Interval()),
      inclusive = "left"
    )}

    hourly_data["T2M"] = hourly_temperature_2m
    hourly_data["RH2M"] = hourly_relative_humidity_2m
    hourly_data["PRECTOTCORR"] = hourly_rain

    hourly_dataframe = pd.DataFrame(data = hourly_data).set_index("ds")
    hourly_dataframe = hourly_dataframe.resample("15min").interpolate(method="linear").round(2)
    hourly_dataframe.index = hourly_dataframe.index - pd.Timedelta(hours=5)

    inicio, fin = pd.Timestamp('2025-05-15 16:15:00'), pd.Timestamp(datetime.now() - pd.Timedelta(hours=5))
    hourly_dataframe = hourly_dataframe[(hourly_dataframe.index >= inicio) & (hourly_dataframe.index <= fin)]
    return hourly_dataframe.reset_index()

def graficar_consumo(df, pron=None, titulo=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"], y=df["value"], mode="lines"))
    if pron is not None:
        fig.add_trace(go.Scatter(x=df["ds"], y=pron, mode="lines"))
    fig.update_layout(title=titulo,
                      xaxis=dict(domain=[0.1, 0.99],title="Fecha", showline=True, linecolor='black', showgrid=False, zeroline=False),
                      yaxis_title="Consumo (kWh)", height=300)
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
        "General": "images/MedidorGen.png",
        "AC": "images/MedidorAA.png",
        "SSFV": "images/MedidorPV.png",
        "Otros": "images/MedidorOtros.png"
    }

# Función para obtener las métricas
def get_metrics(general, ac, ssfv, otros):
    return {
        "General": {"energia": f"{general:.2f}"},
        "AC": {"energia": f"{ac:.2f}"},
        "SSFV": {"energia": f"{-ssfv:.2f}"},
        "Otros": {"energia": f"{otros:.2f}"},
    }

# Submedidores dinámicos
def get_submedidores(metrics):
    return [k for k in metrics if k != "General"]

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
        #width: fit-content!important;
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
        font-size: 1.2em !important;
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
        background-color:#ffffff;
    }
    '''
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

def preparar_futuro(db, datos):
    fut = db[db["unique_id"] == 'General'][['ds', 'value']].rename(columns={'value': 'Energia_kWh_General'})
    fut['ds'] = pd.to_datetime(fut['ds'])
    fut['Hour'] = fut['ds'].dt.hour
    fut['DOW'] = fut['ds'].dt.dayofweek + 1
    #fut['JL'] = ((fut['Hour'].between(8, 16)) & (fut['DOW'].between(1, 5))).astype(int)
    fut = fut.merge(datos.drop(columns='PRECTOTCORR', errors='ignore'), on='ds', how='left')
    return fut[['ds','Energia_kWh_General','Hour','DOW','RH2M','T2M']].sort_values(['ds'])
    #return fut[['ds','Energia_kWh_General','Hour','DOW','RH2M','T2M','JL']].sort_values(['ds'])

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
def display_general(icons, metrics, db):
    colg = st.columns([1, 2, 1])[1]
    with colg:
        with st.container(border=True):
            ca,cb = st.columns([1, 1], vertical_alignment='bottom')
            with ca:
                mostrar_imagen(icons['General'], 100)
            with cb:
                st.metric(label="Energía (kWh)", value=metrics['General']['energia'], delta="100%",delta_color='off')

            if st.button("Ver Detalle", key="butt_gen",use_container_width=True):
                toggle_visibility("vis_gen")
            if st.session_state.get("vis_gen", False):
                df = db.loc[db["unique_id"] == 'General',['ds','value']]
                graficar_consumo(df,None,f"Consumo: Medidor General")


# Función para mostrar los submedidores
def display_submedidores(submedidores, nombres_submedidores, icons, metrics, db, pron):
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
                    df = db.loc[db["unique_id"] == nombre,['ds','value']]
                    graficar_consumo(df, pron[nombre], f"Consumo: {nombre}")


def display_extern_cond(datos, con_boton=False, lat=None, lon=None):
    with st.container(border=True):
        st.subheader("Condiciones Externas",help="")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌡️ Temperatura (°C)", f"{datos['T2M'].iloc[-1]:.2f} ")
        with col2:
            st.metric("💧 Humedad Relativa (%)", f"{datos['RH2M'].iloc[-1]:.2f}")
        with col3:
            st.metric("🌧️ Precipitaciones (mm)", f"{datos['PRECTOTCORR'].iloc[-1]:.2f}")
        if con_boton:
            if st.button("Ver Detalle", key="butt_clima", use_container_width=True):
                toggle_visibility("vis_clima")

            if st.session_state.get("vis_clima", False):
                graficar_cond(get_climate_data_1m(lat, lon))

def display_intern_cond(db):
    df = db.loc[db["unique_id"] == 'General']
    if len(df) >= 2:
        diferencia = (df["value"].iloc[-1] - df["value"].iloc[-2])
    else:
        diferencia = None
    with st.container(border=True):
        st.subheader("Condiciones Internas",help='Condiciones asociadas directamente con el edificio')
        col1, col2, = st.columns(2)
        with col1:
            st.metric("👥 Personas", f"{round(1,0)}")
        with col2:
            st.metric("⚡Consumo (kWh)", f"{round(df['value'].iloc[-1],2)}", delta=f"{round(diferencia,1)} kWh", delta_color="inverse")


def quarter_autorefresh(key: str = "q", state_key: str = "first") -> None:
    """Refresca en el próximo cuarto de hora exacto y luego cada 15 min."""
    ms_to_q = lambda: ((15 - datetime.now().minute % 15) * 60
                       - datetime.now().second) * 1000 \
                      - datetime.now().microsecond // 1000
    first = st.session_state.setdefault(state_key, True)
    interval = ms_to_q() if first else 15 * 60 * 1000
    st.session_state[state_key] = False
    st_autorefresh(interval=interval, key=key)

# Método principal
def main():
    set_page_config()
    quarter_autorefresh()
    icons = get_icons()

    credentials = bigquery_auth()
    db = read_bq_db(credentials)

    lat, lon = 3.4793949016367822, -76.52284557701176
    datos = get_climate_data_1m(lat, lon)
    with st.container(key="styled_tabs"):
        tab1, tab2 = st.tabs(["⚡ Medición Inteligente No Intrusiva ", " Smart Building "])

    nombres_submedidores = {
        "AC": "Aires Acondicionados",
        "SSFV": "SSFV",
        "otros": "Otras Cargas"
    }

    modelo_IA = get_IA_model()
    caracteristicas = preparar_futuro(db, datos).drop(columns=['ds']).values
    caracteristicas = caracteristicas.reshape(-1, 1, caracteristicas.shape[1])
    Y_hat_df2 = pd.DataFrame(modelo_IA.predict(caracteristicas), columns=['Aires Acondicionados','SSFV','Otros'])
    Y_hat_df2.index = db.loc[db["unique_id"] == 'General', "ds"].reset_index(drop=True)
    #st.write(caracteristicas.reshape(caracteristicas.shape[0],caracteristicas.shape[2]))
    #st.write(Y_hat_df2)
    metrics = get_metrics(db.loc[db["unique_id"] == 'General',"value"].iloc[-1],
                          Y_hat_df2['Aires Acondicionados'].iloc[-1],
                          Y_hat_df2['SSFV'].iloc[-1],
                          Y_hat_df2['Otros'].iloc[-1])

    inject_css()

    with tab1.container(key='cont-nilm'):
        display_extern_cond(datos)
        submedidores = get_submedidores(metrics)
        with st.container(border=True):
            st.subheader("Medición Desagregada")
            display_general(icons, metrics, db)
            st.markdown("<div style='text-align:center; margin-top: -20px; font-size:33px;'> ┌──────────────┼──────────────┐<br></div>", unsafe_allow_html=True)
            display_submedidores(submedidores, nombres_submedidores, icons, metrics, db, Y_hat_df2)

    with tab2.container(key='cont-ci'):
        col1, col2 = st.columns([2, 1])
        with col1:
            display_extern_cond(datos,False,lat,lon)
        with col2:
            display_intern_cond(db)

    zona = pytz.timezone("America/Bogota")
    ahora = pd.Timestamp(datetime.now(zona)).floor('15min').strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"🔄 Esta página se actualiza cada 15 minutos. Última actualización: {ahora}")

if __name__ == "__main__":
    main()
