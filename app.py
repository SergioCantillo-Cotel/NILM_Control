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
import requests, pytz, holidays
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components
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
    #st.title("📈 Proyectos IA + Eficiencia energética")

def quarter_autorefresh(key: str = "q", state_key: str = "first") -> None:
    """Refresca en el próximo cuarto de hora exacto y luego cada 15 min."""
    ms_to_q = lambda: ((15 - datetime.now().minute % 15) * 60
                       - datetime.now().second) * 1000 \
                      - datetime.now().microsecond // 1000
    first = st.session_state.setdefault(state_key, True)
    interval = ms_to_q() if first else 15 * 60 * 1000
    st.session_state[state_key] = False
    st_autorefresh(interval=interval, key=key)

def bigquery_auth():
    return service_account.Credentials.from_service_account_info(credenciales_json)

def read_bq_db(credentials):
    # Constantes internas
    _POWER = [
        'PM_General_Potencia_Activa_Total',
        'PM_Aires_Potencia_Activa_Total',
        'Inversor_Solar_Potencia_Salida'
    ]
    _INTERN = ['ocupacion_sede','IDU_0_1_Room_Temperature','IDU_0_2_Room_Temperature','IDU_0_1_Room_Temperature',
               'IDU_0_3_Room_Temperature','IDU_0_4_Room_Temperature','IDU_0_5_Room_Temperature',
               'IDU_0_6_Room_Temperature','IDU_0_7_Room_Temperature','IDU_0_8_Room_Temperature',
               'IDU_0_9_Room_Temperature','IDU_0_10_Room_Temperature']

    _MAPPING = {
        'PM_General_Potencia_Activa_Total': 'General',
        'PM_Aires_Potencia_Activa_Total': 'Aires Acondicionados',
        'Inversor_Solar_Potencia_Salida': 'SSFV',
        'ocupacion_sede': 'Ocupacion',
        'IDU_0_1_Room_Temperature': 'T1','IDU_0_2_Room_Temperature': 'T2',
        'IDU_0_3_Room_Temperature': 'T3','IDU_0_4_Room_Temperature': 'T4',
        'IDU_0_5_Room_Temperature': 'T5','IDU_0_6_Room_Temperature': 'T6',
        'IDU_0_7_Room_Temperature': 'T7','IDU_0_8_Room_Temperature': 'T8',
        'IDU_0_9_Room_Temperature': 'T9','IDU_0_10_Room_Temperature': 'T10'
    }

    # Leer datos de BigQuery
    query = f"SELECT * FROM `{TABLA_COMPLETA}` ORDER BY datetime_record ASC"
    df = (read_gbq(query, project_id=BIGQUERY_PROJECT_ID, credentials=credentials)
          .rename(columns={'id': 'unique_id', 'datetime_record': 'ds'}))

    # Separar en potencia y resto
    df_power = df[df['unique_id'].isin(_POWER)].copy()
    df_other = df[df['unique_id'].isin(_INTERN)].copy()

    # Procesar datos de potencia: escala, tiempo y mapeo
    df_power['value'] *= 0.25
    df_power['ds'] = pd.to_datetime(df_power['ds']).dt.floor('15min')
    df_power['unique_id'] = df_power['unique_id'].map(_MAPPING)
    df_power = gen_others_load(df_power)

    # Procesar el resto: solo mapeo de identificadores
    df_other['unique_id'] = df_other['unique_id'].map(_MAPPING)

    return df_power, df_other

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
      "hourly": ["apparent_temperature", "relative_humidity_2m", "precipitation"],
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

def graficar_consumo(df,pron,sub):
    fig = go.Figure()
    
    if not sub:
        fig.add_trace(go.Bar(x=df["ds"],y=np.where(df['value'] > 0, -pron['SSFV'], - pron["SSFV"] + pron['Otros'] + pron["Aires Acondicionados"]),name="Solar",marker_color="orange"))
        fig.add_trace(go.Bar(x=df["ds"],y=np.where(df['value'] > 0, pron['Otros'], 0), name="Otros",marker_color="gray"))
        fig.add_trace(go.Bar(x=df["ds"],y=np.where(df['value'] > 0, - pron['SSFV'] + pron['Aires Acondicionados'], 0),name="Aires Acondicionados",marker_color="lightblue"))
        fig.add_trace(go.Scatter(x=df["ds"], y=round(df["value"],1), mode="lines",name='General',line=dict(color='black')))
    
    else:
        fig.add_trace(go.Scatter(x=df["ds"], y=df["value"], mode="lines",name='Real'))
        fig.add_trace(go.Scatter(x=df["ds"], y=pron, mode="lines",name='Pronosticado'))
    fig.update_layout(title="", margin=dict(t=30, b=0), barmode="relative",
                      xaxis=dict(domain=[0.1, 0.99],title="Fecha", showline=True, linecolor='black', showgrid=False, zeroline=False),
                      yaxis_title="Consumo (kWh)",legend=dict(orientation="h", x=0.5, xanchor="center", y=1.4, yanchor="top"), height=200)
    st.plotly_chart(fig, use_container_width=True)

def graficar_intensidad_heatmap(ruta_excel):
    df = pd.read_excel(ruta_excel)
    df["dia_semana"] = pd.Categorical(df["dia_semana"], categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], ordered=True)
    tabla = df.pivot(index='dia_semana', columns='hora', values='intensidad')
    custom_text = [[f"Día: {dia}<br>Hora: {hora}<br>Intensidad: {tabla.loc[dia, hora]:.1f}" for hora in tabla.columns] for dia in tabla.index]
    fig = go.Figure(go.Heatmap(z=tabla.values, x=tabla.columns, y=tabla.index,colorscale='Viridis', colorbar_title='Intensidad', text=custom_text, hoverinfo='text', xgap=1, ygap=1))
    fig.update_layout(title='Programación de Intensidad Habitual de Aires', xaxis=dict(title='Hora', showgrid=True), yaxis=dict(title='Día', showgrid=True), template='simple_white')
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
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.4, yanchor="top"), height=200)
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
        "General": {"energia": f"{general:.1f}"},
        "AC": {"energia": f"{ac:.1f}"},
        "SSFV": {"energia": f"{-ssfv:.1f}"},
        "Otros": {"energia": f"{otros:.1f}"},
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

    .block-container {
        padding-top: 1rem;
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
        font-size:2em!important;
        margin: auto;
    }

    div[data-testid="stMetric"] > div:nth-child(2) {
        font-family: 'Poppins';
        font-size:2em!important;
    }

    .stButton {
        display: flex;
        justify-content: center;
        margin-top: 0em;
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

    div[data-testid="metric-container"] > label[data-testid="stMetricDelta"] > div {
			font-size: large;
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
def display_general(icons, metrics, db, pron):
    with st.container(border=True):
        colg,colh = st.columns([1, 2], vertical_alignment='center')
        with colg:
            ca,cb = st.columns([1, 2], vertical_alignment='center')
            with ca:
                mostrar_imagen(icons['General'], 200)
            with cb:
                st.metric(label="Medición General", value=metrics['General']['energia']+" kWh (100%)")

        with colh:
            df = db.loc[db["unique_id"] == 'General',['ds','value']]
            graficar_consumo(df,pron,False)


# Función para mostrar los submedidores
def display_submedidores(submedidores, nombres_submedidores, icons, metrics, db, pron):
    cols = st.columns(len(submedidores))
    for i, label in enumerate(submedidores):
        nombre = nombres_submedidores.get(label, label)
        with cols[i]:
            with st.container(border=True):
                ca,cb = st.columns([1, 2], vertical_alignment='top')
                with ca:
                    mostrar_imagen(icons[label], 100)
                with cb:
                    porc = (pron.iloc[-1,i]/sum(pron.iloc[-1,:]))*100
                    st.metric(label=nombre, value=f"{metrics[label]['energia']} kWh ({porc:.1f}%)")

                key_btn = f"butt_{nombre}"
                key_vis = f"vis_{nombre}"

                if st.button("Ver Detalle", key=key_btn,use_container_width=True):
                    toggle_visibility(key_vis)

                if st.session_state.get(key_vis, False):
                    df = db.loc[db["unique_id"] == nombre,['ds','value']]
                    graficar_consumo(df, pron[nombre], True)


def display_extern_cond(datos, con_boton=False, lat=None, lon=None):
    with st.container(border=True):
        st.markdown("#### Condiciones Externas")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌡️ Temperatura", f"{datos['T2M'].iloc[-1]:.1f} °C")
            st.markdown('<br>',unsafe_allow_html=True)
        with col2:
            st.metric("💧 Hum. Relativa", f"{datos['RH2M'].iloc[-1]:.1f} %")
        with col3:
            st.metric("🌧️ Precipitaciones", f"{datos['PRECTOTCORR'].iloc[-1]:.1f} mm")

def display_intern_cond(db1,db2):
    df = db2.loc[db2["unique_id"] == 'General']
    df_temp = db1.loc[db1["unit"] == '°C']
    promedios = (df_temp.groupby(df_temp['ds'].dt.strftime('%Y-%m-%d %H:%M:%S'))['value'].mean())
    df_pers = db1.loc[db1["unique_id"] == 'Ocupacion']
    if len(df) >= 2:
        diferencia = (df["value"].iloc[-1] - df["value"].iloc[-2])
    else:
        diferencia = None
    with st.container(border=True):
        st.markdown("#### Condiciones Internas")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👥 Personas", f"{df_pers['value'].iloc[-1]:.0f}")
            st.markdown('<br>',unsafe_allow_html=True)
        with col2:
            st.metric("⚡Consumo", f"{round(df['value'].iloc[-1],1)} kWh", delta=f"{round(diferencia,1)} kWh", delta_color="inverse")
        with col3:
            st.metric("🌡️ Temperatura",f"{promedios.iloc[-1]:.1f} °C")
    return promedios.iloc[-1]


def display_smart_control(db1,db2,t_int):
    personas = db1.loc[db1["unique_id"] == 'Ocupacion'].iloc[-1,2]
    t_ext = db2['T2M'].iloc[-1]
    with st.container(border=True):
         st.markdown("#### Control Edificio")
         with st.container(key="styled_tabs_2"):
              tab1, tab2 = st.tabs(["Programación Estándar", "Programación IA"])
              with tab1.container(key='cont-BMS'):
                  graficar_intensidad_heatmap("BMS/programacion_bms.xlsx")

              with tab2.container(key='cont-BMS-IA'):
                  ruta = 'BMS/programacion_bms.xlsx'
                  resultado, pronostico, base = agenda_bms(ruta,datetime.now()-pd.Timedelta(hours=5),personas,t_ext,t_int)
                  st.error(resultado)
                  st.warning(resultado)
                  st.success(resultado)
                  unidades = seleccionar_unidades(pronostico, base)
                  #st.write(unidades)
                  cols = st.columns(5)
                  estados = {}
                  for i, col in enumerate(cols):
                      with col:
                          # si el valor es 1 o 2, lo marcamos como True
                          inicial = unidades[i] == 1
                          estados[f"aire_{i+1}"] = st.toggle(f"Aire {i+1}", value=inicial, key=f"aire_{i+1}")
                  
                  file_ = open("Piso-1-lado-A-aires-Encendidos.gif", "rb")
                  contents = file_.read()
                  data_url = base64.b64encode(contents).decode("utf-8")
                  file_.close()

                  st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',unsafe_allow_html=True)
                  col1, col2, col3 = st.columns(3)
                  with col1:
                      st.markdown('<div style="text-align: center;"><a href="http://192.168.5.200:3000/" target="_blank">Piso 1: Lado A</a></div>',unsafe_allow_html=True)
                  with col2:
                      st.markdown('<div style="text-align: center;"><a href="http://192.168.5.200:3000/Piso_2" target="_blank">Piso 2: Lado A</a></div>',unsafe_allow_html=True)
                  with col3:
                      st.markdown('<div style="text-align: center;"><a href="http://192.168.5.200:3000/Piso_1_Lado_B" target="_blank">Piso 1: Lado B</a></div>',unsafe_allow_html=True)
                  

def agenda_bms(ruta, fecha, num_personas, temp_ext, temp_int):
    df = pd.read_excel(ruta, usecols=[0, 1, 2, 3], names=['dia', 'hora', '_', 'intensidad'])
    dia_str = fecha.strftime('%A')
    dias_es = {'Monday': 'Lunes','Tuesday': 'Martes','Wednesday': 'Miércoles',
               'Thursday': 'Jueves','Friday': 'Viernes','Saturday': 'Sábado',
               'Sunday': 'Domingo'}
    dia_S = dias_es[dia_str]
    h = fecha.hour
    festivos = holidays.CountryHoliday('CO', years=fecha.year)
    
    if fecha.date() in festivos:
        return f"Hoy es {dia_str} y la hora actual es {h}, la programación Estándar del BMS es: Intensidad de Aires 0 %"
    
    base = df.query("dia == @dia_str and hora == @h")['intensidad']
    if base.empty:
        return f"No hay programación registrada para {dia_str} a las {h}:00 horas."
    
    b = base.iat[0]
    ajuste_personas = (-100 if num_personas < 5 else
                       -50 if num_personas < 10 else
                       -25 if num_personas < 20 else
                       0 if num_personas < 40 else
                       25 if num_personas < 50 else 50)
    
    p = max(0, min(100, b - (25 - temp_ext) + 1.5 * (temp_int - 25) + ajuste_personas))
    delta = p - b

    categoria = (1 if delta < -10 else
                 2 if delta < -5 else
                 4 if delta <= 5 else
                 6 if delta <= 10 else 7)

    return (f"Hoy, {dia_S} a las {h}:{fecha.minute:02}, la programación Estándar del BMS indica que la Intensidad de Aires esté al {b}%.\n"
            f"Ahora, dado que hay {num_personas:.0f} personas en la sede, temperaturas externa e interna de {temp_ext:.1f} °C y {temp_int:.1f} °C respectivamente, "
            f"el modelo IA sugiere una intensidad de {p:.0f}% con una velocidad de ventiladores de {categoria}"), p, b

def seleccionar_unidades(pred, intensidad_base):
    tabla_intensidad = {
        0:  [0, 0, 0, 0, 0],
        15: [1, 0, 0, 0, 0],
        25: [1, 0, 1, 0, 0],
        50: [1, 0, 1, 1, 0],
        75: [1, 0, 1, 1, 1],
        100:[1, 1, 1, 1, 1],
    }

    diferencia = abs(pred - intensidad_base)
    intensidades = sorted(tabla_intensidad.keys())

    if diferencia < 10:
        intensidad_cercana = max([i for i in intensidades if i <= intensidad_base], default=0)
    else:
        intensidad_cercana = max([i for i in intensidades if i <= pred], default=0)

    return tabla_intensidad[intensidad_cercana]


def get_IA_model():
    IA_model = load_model('models/NILM_Model_best.keras')
    #IA_model = load_model('models/NILM_Model.keras')
    return IA_model

def datos_Exog(db, datos):
    fut = db[db["unique_id"] == 'General'][['ds', 'value']].rename(columns={'value': 'Energia_kWh_General'})
    fut['ds'] = pd.to_datetime(fut['ds'])
    fut['DOW'] = fut['ds'].dt.dayofweek + 1
    fut['Hour'] = fut['ds'].dt.hour
    fut = fut.merge(datos.drop(columns='PRECTOTCORR', errors='ignore'), on='ds', how='left')
    return fut[['ds','Energia_kWh_General','DOW','Hour','T2M','RH2M']].sort_values(['ds'])

def reconcile(exog, pron):
    r = np.copy(pron)
    d = round(exog['Energia_kWh_General'],1) - (r[:,0] - r[:,1] + r[:,2])
    for i in range(len(r)):
        dow, h, di = exog['DOW'][i], exog['Hour'][i], r[i]
        wknd, work, sun, dia = dow in (6,7), 8<=h<=16, 11<=h<=13, 6<=h<=18

        if not dia:
            di[1], di[2] = 0, di[2] + d[i]
        elif wknd:
            di[2] += d[i]
        elif work:
            if sun:
                adj = min(d[i], di[1])
                di[1] -= adj
                di[2] += d[i] - adj
            else:
                adj = min(d[i]*0.5, di[1])
                di[1] -= adj
                di[2] += d[i] - adj
        else:
            adj = min(d[i], di[1])
            di[1] -= adj
            di[2] += d[i] - adj

        if dia and not wknd and not work and not sun:
            t = di[1] + di[2]
            if t > 0:
                di[1] += d[i] * di[1]/t
                di[2] += d[i] * di[2]/t
            else:
                di[1] += d[i]*0.5
                di[2] += d[i]*0.5

        di[1] = max(di[1], 0)
        di[2] = max(di[2], 0)

    return r

# Método principal
def main():
    set_page_config()
    quarter_autorefresh()
    icons = get_icons()

    credentials = bigquery_auth()
    db_pow, db_oth = read_bq_db(credentials)
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
    caracteristicas = datos_Exog(db_pow, datos).drop(columns=['ds'])
    car2 = caracteristicas.copy()
    y_hat_raw = modelo_IA.predict(caracteristicas.values.reshape(-1, 1, caracteristicas.shape[1]))
    Y_hat_rec = reconcile(car2,y_hat_raw)
    Y_hat_df2 = pd.DataFrame(Y_hat_rec, columns=['Aires Acondicionados','SSFV','Otros'])

    Y_hat_df2.index = db_pow.loc[db_pow["unique_id"] == 'General', "ds"].reset_index(drop=True)
    metrics = get_metrics(db_pow.loc[db_pow["unique_id"] == 'General',"value"].iloc[-1],
                          Y_hat_df2['Aires Acondicionados'].iloc[-1],
                          Y_hat_df2['SSFV'].iloc[-1],
                          Y_hat_df2['Otros'].iloc[-1])

    inject_css()

    with tab1.container(key='cont-nilm'):
        #display_extern_cond(datos)
        submedidores = get_submedidores(metrics)
        with st.container(border=True):
            display_general(icons, metrics, db_pow, Y_hat_df2)
            st.markdown("<div style='text-align:center; margin-top: -20px; font-size:33px;'> ┌──────────────┼──────────────┐<br></div>", unsafe_allow_html=True)
            display_submedidores(submedidores, nombres_submedidores, icons, metrics, db_pow, Y_hat_df2)

    with tab2.container(key='cont-ci'):
        col1, col2 = st.columns([1, 1], vertical_alignment='bottom')
        with col1:
            display_extern_cond(datos,False,lat,lon)
        with col2:
            prom = display_intern_cond(db_oth,db_pow)

        display_smart_control(db_oth,datos,prom)

    zona = pytz.timezone("America/Bogota")
    ahora = pd.Timestamp(datetime.now(zona)).floor('15min').strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"🔄 Esta página se actualiza cada 15 minutos. Última actualización: {ahora}")

if __name__ == "__main__":
    main()
