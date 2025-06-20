import streamlit as st
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import requests_cache, holidays, openmeteo_requests, math
from retry_requests import retry
from google.oauth2 import service_account
from pandas_gbq import read_gbq
import pandas as pd
import numpy as np
import xgboost as xgb

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
    _TEMP = [f'IDU_0_{i}_Room_Temperature' for i in range(1, 11)]+[f'IDU_0_{i}_Estado_valvula' for i in range(1, 11)]+['ocupacion_sede']
    _OCCUP = ['ocupacion_UMA_1','ocupacion_UMA_2','ocupacion_UMA_3','ocupacion_UMA_4',
              'ocupacion_UMA_5','ocupacion_flotante'] 
    _MAPPING = {'PM_General_Potencia_Activa_Total': 'General','PM_Aires_Potencia_Activa_Total': 'Aires Acondicionados',
                'Inversor_Solar_Potencia_Salida': 'SSFV','ocupacion_sede': 'Ocupacion',
                'ocupacion_flotante':'Flotantes',
                **{f'IDU_0_{i}_Room_Temperature': f'T{i}' for i in range(1, 11)},
                **{f'IDU_0_{i}_Estado_valvula': f'Valvula_{i}' for i in range(1, 11)},
                'ocupacion_UMA_1':'Z1: Sala de Juntas - Cubiculos',
                'ocupacion_UMA_2':'Z2: Gerencia - Area TI',
                'ocupacion_UMA_3':'Z3: G. Humana - EE - Preventa',
                'ocupacion_UMA_4':'Z4: Contabilidad - Sala de Juntas',
                'ocupacion_UMA_5':'Z5: G. Humana - Depto. Jurídico'}
    df = read_gbq(f"SELECT * FROM `{TABLA_COMPLETA}` ORDER BY datetime_record ASC, id ASC", project_id=BIGQUERY_PROJECT_ID, credentials=credentials).rename(columns={'id': 'unique_id', 'datetime_record': 'ds'})
    #df['ds'] = pd.to_datetime(df['ds']).dt.floor('15min')
    df_power = df[df['unique_id'].isin(_POWER)].copy()
    df_power['value'] *= 0.25
    df_power['ds'] = pd.to_datetime(df_power['ds']).dt.floor('15min')
    df_power['unique_id'] = df_power['unique_id'].map(_MAPPING)
    df_power = gen_others_load(df_power)
    
    df_AC = df[df['unique_id'].isin(_TEMP)].copy()
    df_AC['unique_id'] = df_AC['unique_id'].map(_MAPPING)
    
    df_occup = df[df['unique_id'].isin(_OCCUP)]
    df_occup['unique_id'] = df_occup['unique_id'].map(_MAPPING) 
    return df_power, df_AC, df_occup

def gen_others_load(df):
    pivot = (df.pivot_table(index=['ds', 'unit', 'company', 'headquarters'],columns='unique_id',values='value').reset_index())
    pivot['Otros'] = pivot['General'] + pivot['SSFV'] - pivot['Aires Acondicionados']
    result = (pivot.melt(id_vars=['ds', 'unit', 'company', 'headquarters'],value_vars=['General', 'Aires Acondicionados', 'SSFV', 'Otros'],var_name='unique_id',value_name='value').sort_values(['ds', 'unique_id']).reset_index(drop=True))
    return result

def get_climate_data(lat, lon):
    session = retry(requests_cache.CachedSession('.cache', expire_after=3600), retries=5, backoff_factor=0.2)
    r = openmeteo_requests.Client(session=session).weather_api("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon, "models": "gfs_seamless",
        "minutely_15": ["temperature_2m", "relative_humidity_2m", "precipitation"],
        "start_date": "2025-05-15", "end_date": datetime.now().strftime("%Y-%m-%d")})[0].Minutely15()
    idx = pd.date_range(start=pd.to_datetime(r.Time(), unit="s"), end=pd.to_datetime(r.TimeEnd(), unit="s"),
                        freq=pd.Timedelta(seconds=r.Interval()), inclusive="left")
    df = pd.DataFrame({"ds": idx, "T2M": r.Variables(0).ValuesAsNumpy(), "RH2M": r.Variables(1).ValuesAsNumpy(),
                       "PRECTOTCORR": r.Variables(2).ValuesAsNumpy()}).set_index("ds")
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

def get_temp_prom(db1):
    df_temp = db1.loc[db1["unit"] == '°C']
    promedios = (df_temp.groupby(df_temp['ds'].dt.strftime('%Y-%m-%d %H:%M:%S'))['value'].mean())
    return promedios

def digital_twin(entradas_DT):
    entradas_DT_d = entradas_DT.drop(columns='ds')
    booster = xgb.Booster()
    booster.load_model("IA\modelo_xgb.model")
    dtest = xgb.DMatrix(entradas_DT_d)
    DT = booster.predict(dtest)
    fechas = entradas_DT['ds'].values    
    DT = pd.DataFrame({'ds': fechas,'Dig_Twin': DT})
    return DT

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

    pron = max(0, min(100, b - (25 - temp_ext) + 1.5 * (temp_int - 25) + ajuste_personas))
    return dia_S, pron

def nueva_carga(pred, personas):
    zonas = personas[(personas["ds"] == personas["ds"].max()) & (personas["unique_id"] != 'Flotantes')]
    zonas = zonas.drop_duplicates(subset=['ds', 'unique_id'], keep='first')
    if zonas['value'].sum() != 0:
        aires = math.ceil(pred / 20)
        zonas['capacidad'] = np.array([6, 21, 26, 30, 15])
        zonas['proporcion_ocup'] = zonas['value'] / zonas['value'].sum()
        zonas['disponibilidad'] = round(100 - (zonas['value'] / zonas['capacidad'] * 100), 2)
        zonas = zonas.sort_values(by='proporcion_ocup', ascending=False)
    else:
        aires = 0
        zonas['proporcion_ocup'] = 0
        zonas['disponibilidad'] = 100

    base = np.array([20, 40, 60, 80, 100])
    zonas['encendido'] = 0
    zonas.loc[zonas.head(int(aires)).index.intersection(zonas[zonas['disponibilidad'] < 100].index), 'encendido'] = 1
    no_encendidos = (zonas['encendido'] != 1).sum()
    carga = max(0, pred - (20 * no_encendidos))  
    return carga

def seleccionar_unidades(pred,personas,fecha,dia):
    zonas = personas[(personas["ds"] == personas["ds"].max()) & (personas["unique_id"] != 'Flotantes')]
    if zonas['value'].sum() != 0:
        # Primer cálculo provisional (no se usa para encendido)
        zonas['capacidad'] = np.array([6, 21, 26, 30, 15])
        zonas['proporcion_ocup'] = zonas['value'] / zonas['value'].sum()
        zonas['disponibilidad'] = round(100 - (zonas['value'] / zonas['capacidad'] * 100), 2)
        zonas = zonas.sort_values(by='proporcion_ocup', ascending=False)
    else:
        zonas['proporcion_ocup'] = 0
        zonas['disponibilidad'] = 100
    
    base = np.array([20, 40, 60, 80, 100])
    # Ajuste de carga real
    no_encendidos = (zonas['disponibilidad'] == 100).sum()
    carga = max(0, pred - (20 * no_encendidos))
    aires = math.ceil(carga / 20) if carga > 0 else 0

    zonas['encendido'] = 0
    if aires > 0:
        zonas.loc[zonas.head(int(aires)).index.intersection(zonas[zonas['disponibilidad'] < 100].index), 'encendido'] = 1

    diferencia = carga - base
    velocidad_valor = np.clip(np.where(diferencia >= 0, 7, np.ceil(((diferencia + 20) / 20) * 7)), 0, 7)
    zonas['velocidad_ventilador'] = zonas['encendido'] * velocidad_valor
    zonas = zonas.sort_values(by='unique_id', ascending=True)   
    
    if zonas['encendido'].sum() == 0:
        mensaje = (
            f"Cotel IA sugiere en este momento, {dia} a las {fecha.hour}:{fecha.strftime('%M')}, "
            "que los aires acondicionados en las distintas zonas estén apagados de acuerdo con las condiciones de control."
        )
        encendidas = [0, 0, 0, 0, 0]
    else:
        encendidas = zonas['encendido'].values.tolist()
        zonas_encendidas = zonas[zonas['encendido'] == 1][['unique_id', 'capacidad', 'value']]
        if not zonas_encendidas.empty:
            zonas_lista = (
                "\n" +
                '\n'.join(
                    [
                        f"- {uid}: {100 * (cap - ocup) / cap:.2f}% de capacidad disponible"
                        for uid, ocup, cap in zip(zonas_encendidas['unique_id'], zonas_encendidas['value'], zonas_encendidas['capacidad'])
                    ]
                )
            )
        else:
            zonas_lista = "(No hay zonas encendidas)"
        mensaje = (
            f"Cotel IA sugiere en este momento, {dia} a las {fecha.hour}:{fecha.strftime('%M')}, "
            "que sean encendidos los aires en las siguientes zonas de acuerdo con las condiciones de control:"
            f"{zonas_lista}\n\n"
            "Invitamos a las personas que se encuentren en otros espacios y quieran disfrutar de un mayor confort "
            "a ubicarse en estos lugares y disfruten de la compañía de la familia Cotel."
        )
    return encendidas, zonas['velocidad_ventilador'].values.tolist(), mensaje, carga
