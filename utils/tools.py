import streamlit as st
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import requests_cache, holidays, openmeteo_requests
from retry_requests import retry
from google.oauth2 import service_account
from pandas_gbq import read_gbq
import pandas as pd

credenciales_json = st.secrets["gcp_service_account"]
BIGQUERY_PROJECT_ID = st.secrets["bigquery"]["project_id"]
BIGQUERY_DATASET_ID = st.secrets["bigquery"]["dataset_id"]
TABLA = st.secrets["bigquery"]["table"]
TABLA_COMPLETA = f"{BIGQUERY_DATASET_ID}.{TABLA}"

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
    _POWER = ['PM_General_Potencia_Activa_Total', 'PM_Aires_Potencia_Activa_Total', 'Inversor_Solar_Potencia_Salida']
    _INTERN = ['ocupacion_sede','IDU_0_1_Room_Temperature','IDU_0_2_Room_Temperature','IDU_0_1_Room_Temperature',
               'IDU_0_3_Room_Temperature','IDU_0_4_Room_Temperature','IDU_0_5_Room_Temperature',
               'IDU_0_6_Room_Temperature','IDU_0_7_Room_Temperature','IDU_0_8_Room_Temperature',
               'IDU_0_9_Room_Temperature','IDU_0_10_Room_Temperature']
    _MAPPING = {'PM_General_Potencia_Activa_Total': 'General','PM_Aires_Potencia_Activa_Total': 'Aires Acondicionados',
                'Inversor_Solar_Potencia_Salida': 'SSFV','ocupacion_sede': 'Ocupacion',
                **{f'IDU_0_{i}_Room_Temperature': f'T{i}' for i in range(1, 11)}}
    df = read_gbq(f"SELECT * FROM `{TABLA_COMPLETA}` ORDER BY datetime_record ASC", project_id=BIGQUERY_PROJECT_ID, credentials=credentials).rename(columns={'id': 'unique_id', 'datetime_record': 'ds'})
    df_power = df[df['unique_id'].isin(_POWER)].copy()
    df_power['value'] *= 0.25
    df_power['ds'] = pd.to_datetime(df_power['ds']).dt.floor('15min')
    df_power['unique_id'] = df_power['unique_id'].map(_MAPPING)
    df_power = gen_others_load(df_power)
    df_other = df[df['unique_id'].isin(_INTERN)].copy()
    df_other['unique_id'] = df_other['unique_id'].map(_MAPPING)
    return df_power, df_other

def gen_others_load(df):
    pivot = (df.pivot_table(index=['ds', 'unit', 'company', 'headquarters'],columns='unique_id',values='value').reset_index())
    pivot['Otros'] = pivot['General'] + pivot['SSFV'] - pivot['Aires Acondicionados']
    result = (pivot.melt(id_vars=['ds', 'unit', 'company', 'headquarters'],value_vars=['General', 'Aires Acondicionados', 'SSFV', 'Otros'],var_name='unique_id',value_name='value').sort_values(['ds', 'unique_id']).reset_index(drop=True))
    return result

def get_climate_data_1m(lat, lon):
    session = retry(requests_cache.CachedSession('.cache', expire_after=3600), retries=5, backoff_factor=0.2)
    r = openmeteo_requests.Client(session=session).weather_api("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon,
        "hourly": ["apparent_temperature", "relative_humidity_2m", "precipitation"],
        "start_date": "2025-05-15", "end_date": datetime.now().strftime("%Y-%m-%d")
    })[0].Hourly()
    idx = pd.date_range(start=pd.to_datetime(r.Time(), unit="s"), end=pd.to_datetime(r.TimeEnd(), unit="s"),
                        freq=pd.Timedelta(seconds=r.Interval()), inclusive="left")
    df = pd.DataFrame({"ds": idx, "T2M": r.Variables(0).ValuesAsNumpy(), "RH2M": r.Variables(1).ValuesAsNumpy(),
                       "PRECTOTCORR": r.Variables(2).ValuesAsNumpy()}).set_index("ds").resample("15min").interpolate("linear").round(2)
    df.index -= pd.Timedelta(hours=5)
    return df.loc["2025-05-15 16:15:00":datetime.now() - pd.Timedelta(hours=5)].reset_index()

# Función para obtener las métricas
def get_metrics(general, ac, ssfv, otros):
    return {
        "General": {"energia": f"{general:.1f}"},
        "AC": {"energia": f"{ac:.1f}"},
        "SSFV": {"energia": f"{-ssfv:.1f}"},
        "Otros": {"energia": f"{otros:.1f}"},
    }
def get_submedidores(metrics):
    return [k for k in metrics if k != "General"]

def load_custom_css(file_path: str = "styles/style.css"):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
            f"el modelo IA sugiere una intensidad de {p:.0f}% con una velocidad de ventiladores de {categoria}"), p, b, categoria

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
