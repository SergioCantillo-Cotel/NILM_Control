import streamlit as st
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import requests_cache, holidays, openmeteo_requests, math, pyarrow, gc
from retry_requests import retry
from google.oauth2 import service_account
import pandas as pd
import numpy as np
import xgboost as xgb
from pandas_gbq import read_gbq

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

def _get_mapping():
    return {
        'PM_General_Potencia_Activa_Total': 'General',
        'PM_Aires_Potencia_Activa_Total': 'Aires Acondicionados',
        'Inversor_Solar_Potencia_Salida': 'SSFV',
        'ocupacion_sede': 'Ocupacion',
        'ocupacion_flotante': 'Flotantes',
        **{f'IDU_0_{i}_Room_Temperature': f'T{i}' for i in range(1, 11)},
        **{f'IDU_0_{i}_Estado_valvula': f'Valvula_{i}' for i in range(1, 11)},
        'ocupacion_UMA_1': 'Z1: Sala de Juntas - Cubiculos',
        'ocupacion_UMA_2': 'Z2: Gerencia - Area TI',
        'ocupacion_UMA_3': 'Z3: G. Humana - EE - Preventa',
        'ocupacion_UMA_4': 'Z4: Contabilidad - Sala de Juntas',
        'ocupacion_UMA_5': 'Z5: G. Humana - Depto. Jurídico'
    }

def _optimize_types(df):
    for col in ['unique_id', 'unit', 'company', 'headquarters']:
        if col in df.columns:
            df[col] = df[col].astype('string[pyarrow]')
    if 'value' in df.columns:
        df['value'] = df['value'].astype('float32')
    return df

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
    df_temp = db1.loc[db1["unit"] == '°C', ['ds', 'value']].copy()
    df_temp['ds'] = pd.to_datetime(df_temp['ds']).dt.floor('15min')
    df_temp['value'] = df_temp['value'].astype('float32')
    prom = df_temp.groupby('ds', observed=True)['value'].mean()
    del df_temp
    gc.collect()
    return prom

def digital_twin(entradas_DT):
    entradas_DT_d = entradas_DT.drop(columns='ds')
    booster = xgb.Booster()
    booster.load_model("./IA/modelo_xgb.model")
    dtest = xgb.DMatrix(entradas_DT_d)
    DT = booster.predict(dtest)
    fechas = entradas_DT['ds'].values
    DT = pd.DataFrame({'ds': fechas, 'Dig_Twin': DT})
    del entradas_DT, dtest, booster
    gc.collect()
    return DT

def get_prog_bms(inicio, now):
    ruta = 'BMS/Prog_BMS.xlsx'
    festivos = holidays.CountryHoliday('CO', years=range(inicio.year, now.year + 1))
    df_prog = pd.read_excel(ruta, sheet_name='Raw')
    if 'promedio' in df_prog.columns:
        df_prog = df_prog.drop('promedio', axis=1)
    if 'MIN' in df_prog.columns:
        df_prog["hora_min"] = df_prog["HORA"].astype(str).str.zfill(2) + ":" + df_prog["MIN"].astype(str).str.zfill(2)
    else:
        df_prog["hora_min"] = df_prog["HORA"].astype(str).str.zfill(2) + ":00"
    # Optimiza tipos
    for col in ['DIA_SEMANA']:
        if col in df_prog.columns:
            df_prog[col] = df_prog[col].astype('string[pyarrow]')
    if 'hora_min' in df_prog.columns:
        df_prog['hora_min'] = df_prog['hora_min'].astype('string[pyarrow]')
    sch_BMS = (
        pd.DataFrame({'ds': pd.date_range(inicio, now, freq='15min')})
        .assign(
            DIA_SEMANA=lambda x: x.ds.dt.day_name(),
            hora=lambda x: x.ds.dt.hour,
            minuto=lambda x: x.ds.dt.minute,
            fecha=lambda x: x.ds.dt.date,
            hora_min=lambda x: x.ds.dt.strftime('%H:%M')
        )
        .merge(df_prog, on=['DIA_SEMANA', 'hora_min'], how='left')
    )
    sch_BMS['INTENSIDAD'] = np.where(
        sch_BMS['fecha'].isin(festivos), 0, sch_BMS['INTENSIDAD']
    )
    sch_BMS = sch_BMS.drop(columns=['fecha'])
    del df_prog
    gc.collect()
    return sch_BMS

def agenda_bms(ruta, fecha, num_personas, temp_ext, temp_int):
    df = pd.read_excel(ruta, sheet_name='Raw')
    if 'MIN' in df.columns:
        df["hora_min"] = df["HORA"].astype(str).str.zfill(2) + ":" + df["MIN"].astype(str).str.zfill(2)
    else:
        df["hora_min"] = df["HORA"].astype(str).str.zfill(2) + ":00"
    # Optimiza tipos
    if 'DIA_SEMANA' in df.columns:
        df['DIA_SEMANA'] = df['DIA_SEMANA'].astype('string[pyarrow]')
    if 'hora_min' in df.columns:
        df['hora_min'] = df['hora_min'].astype('string[pyarrow]')
    dias_es = {
        "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
        "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"
    }
    if df["DIA_SEMANA"].iloc[0] in dias_es:
        df["DIA_SEMANA"] = df["DIA_SEMANA"].map(dias_es)
    dia_S = dias_es[fecha.strftime('%A')]
    h, m = fecha.hour, fecha.minute

    festivos = holidays.CountryHoliday('CO', years=fecha.year)
    if fecha.date() in festivos:
        del df
        gc.collect()
        return dia_S, 0

    hora_min_actual = f"{str(h).zfill(2)}:{str(m).zfill(2)}"
    base = df.query("DIA_SEMANA == @dia_S and hora_min == @hora_min_actual")['INTENSIDAD']
    if base.empty:
        base = df.query("DIA_SEMANA == @dia_S and HORA == @h")['INTENSIDAD']
        if base.empty:
            del df
            gc.collect()
            return dia_S, 0

    b = base.iat[0]
    if any(pd.isna(x) or x is None for x in [num_personas, temp_int]):
        pron = b
    else:
        ajuste_personas = (
            -100 if num_personas < 5 else
            -50 if num_personas < 10 else
            -25 if num_personas < 20 else
            0 if num_personas < 40 else
            25 if num_personas < 50 else 50
        )
        pron = max(0, min(100, b - (25 - temp_ext) + 1.5 * (temp_int - 25) + ajuste_personas))
    del df
    gc.collect()
    return dia_S, pron

def nueva_carga(pred, personas):
    zonas = personas[(personas["ds"] == personas["ds"].max()) & (personas["unique_id"] != 'Flotantes')].drop_duplicates(subset=["ds", "unique_id"])
    if 'unique_id' in zonas.columns:
        zonas['unique_id'] = zonas['unique_id'].astype('string[pyarrow]')
    if zonas['value'].sum() != 0:
        # Primer cálculo provisional (no se usa para encendido)
        zonas['capacidad'] = np.array([6, 21, 26, 30, 15], dtype='int8')
        zonas['proporcion_ocup'] = zonas['value'] / zonas['value'].sum()
        zonas['disponibilidad'] = round(100 - (zonas['value'] / zonas['capacidad'] * 100), 2)
        zonas = zonas.sort_values(by='proporcion_ocup', ascending=False)
    else:
        zonas['proporcion_ocup'] = 0
        zonas['disponibilidad'] = 100

    aires_ini = math.ceil(pred / 20) if pred > 0 else 0
    zonas_aire = zonas.head(int(aires_ini)).copy()
    no_encendidos = (zonas_aire['disponibilidad'] == 100).sum()
    carga_bruta = max(0, pred - (20 * no_encendidos))
    carga = min(100, int(carga_bruta // 20) * 20)
    #aires = math.ceil(carga / 20) if carga > 0 else 0    
    #print("carga:", carga, "aires", aires)
    del zonas, zonas_aire
    gc.collect()
    return carga

def seleccionar_unidades(pred, personas, fecha, dia):
    zonas = personas[(personas["ds"] == personas["ds"].max()) & (personas["unique_id"] != 'Flotantes')].copy()
    if 'unique_id' in zonas.columns:
        zonas['unique_id'] = zonas['unique_id'].astype('string[pyarrow]')
    if zonas['value'].sum() != 0:
        zonas['capacidad'] = np.array([6, 21, 26, 30, 15], dtype='int8')
        zonas['proporcion_ocup'] = zonas['value'] / zonas['value'].sum()
        zonas['disponibilidad'] = np.round(100 - (zonas['value'] / zonas['capacidad'] * 100), 2)
        zonas = zonas.sort_values(by='proporcion_ocup', ascending=False)
    else:
        zonas['proporcion_ocup'] = 0
        zonas['disponibilidad'] = 100

    aires_ini = math.ceil(pred / 20) if pred > 0 else 0
    zonas_aire = zonas.head(int(aires_ini)).copy()
    no_encendidos = (zonas_aire['disponibilidad'] == 100).sum()
    carga = round(max(0, pred - (20 * no_encendidos)), 2)
    aires = math.ceil(carga / 20) if carga > 0 else 0

    zonas['encendido'] = 0
    if aires > 0:
        idx = zonas.head(int(aires)).index.intersection(zonas[zonas['disponibilidad'] < 100].index)
        zonas.loc[idx, 'encendido'] = 1

    velocidad_valor = np.ceil(7 * (1 - zonas['disponibilidad'] / 100))
    velocidad_valor = np.clip(velocidad_valor, 0, 7).astype(int)
    zonas['velocidad_ventilador'] = (zonas['encendido'] * velocidad_valor).astype(int)
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
    del zonas_aire
    gc.collect()
    return encendidas, zonas['velocidad_ventilador'].values.tolist(), mensaje, carga

def obtener_ia(row, ruta, pz):
    _, pron = agenda_bms(ruta, row['ds'], row['value'], row['T2M'], row['promedio_T'])
    personas_zona = pz[(pz['ds'] == row['ds']) & (pz['unique_id'] != 'Flotantes')]
    return pd.Series({'intensidad_IA': nueva_carga(pron, personas_zona)}, dtype='float32')

def calcular_comparativa(db_AA, db_pers, db_t_ext, db_t_int):
    ruta = 'BMS/Prog_BMS.xlsx'
    now = (pd.Timestamp.now() - pd.Timedelta(hours=5)).floor('15min')
    inicio = now - pd.Timedelta(weeks=1)
    sch_BMS = get_prog_bms(inicio, now)
    fechas_base = sch_BMS['ds']

    db_AA = db_AA.copy(); db_AA['ds'] = db_AA['ds'].dt.floor('15min')
    db_AA = db_AA[(db_AA['ds'] >= inicio) & (db_AA['ds'] <= now)]
    sch_RT = fechas_base.to_frame().merge(
        db_AA.groupby('ds')['value'].agg(lambda x: x.sum() * 100 / x.count()).reset_index(), on='ds', how='left'
    )

    db_pers = db_pers.copy(); db_pers['ds'] = db_pers['ds'].dt.floor('15min')
    db_pers = db_pers[(db_pers['ds'] >= inicio) & (db_pers['ds'] <= now)]
    pz = db_pers[db_pers["ds"].notna()].copy()
    db_pers = fechas_base.to_frame().merge(
        db_pers.groupby('ds')['value'].sum().reset_index(), on='ds', how='left'
    )

    db_t_ext = db_t_ext.copy(); db_t_ext['ds'] = db_t_ext['ds'].dt.floor('15min')
    db_t_ext = db_t_ext[(db_t_ext['ds'] >= inicio) & (db_t_ext['ds'] <= now)]
    db_t_ext = fechas_base.to_frame().merge(
        db_t_ext.groupby('ds')['T2M'].sum().reset_index(), on='ds', how='left'
    )

    db_t_int = db_t_int.copy()
    db_t_int = db_t_int[(db_t_int['ds'] >= inicio) & (db_t_int['ds'] <= now) & (db_t_int['unique_id'].str.match(r'^T(10|[1-9])$'))]
    db_t_int['ds'] = db_t_int['ds'].dt.floor('15min')
    db_t_int = fechas_base.to_frame().merge(
        db_t_int.pivot_table(index='ds', columns='unique_id', values='value', aggfunc='mean').mean(axis=1).reset_index(name='promedio_T'),
        on='ds', how='left'
    )

    df_ia = db_pers[['ds', 'value']].merge(db_t_ext[['ds', 'T2M']], on='ds', how='left').merge(db_t_int[['ds', 'promedio_T']], on='ds', how='left')
    sch_IA = pd.concat([df_ia[['ds']], df_ia.apply(lambda row: obtener_ia(row, ruta, pz), axis=1)], axis=1)

    dif_BMS_RT = pd.Series(np.where(sch_BMS['INTENSIDAD'] != 0, 100 * (sch_BMS['INTENSIDAD'] - sch_RT['value']) / sch_BMS['INTENSIDAD'], np.nan)).fillna(0)
    dif_BMS_IA = pd.Series(np.where(sch_BMS['INTENSIDAD'] != 0, 100 * (sch_BMS['INTENSIDAD'] - sch_IA['intensidad_IA']) / sch_BMS['INTENSIDAD'], np.nan)).fillna(0)

    del db_AA, db_pers, db_t_ext, db_t_int
    gc.collect()
    return sch_BMS, sch_RT, sch_IA, np.nanmean(dif_BMS_RT), np.nanmean(dif_BMS_IA)
