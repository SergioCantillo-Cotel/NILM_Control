import streamlit as st
import base64
from datetime import datetime
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from utils import tools

def toggle_visibility(key):
    if key not in st.session_state:
        st.session_state[key] = True
    else:
        st.session_state[key] = not st.session_state[key]

def graficar_consumo(df,pron,sub):
    fig = go.Figure()

    if not sub:
        fig.add_trace(go.Bar(x=df["ds"], y = np.where(df['value'] > 0, np.where(pron['SSFV'].values < df['value'].values, pron["SSFV"], df['value']), - pron["SSFV"] + pron['Otros'] + pron["Aires Acondicionados"]),name="Solar",marker_color="orange"))
        #fig.add_trace(go.Bar(x=df["ds"], y = np.where(df['value'] > 0, pron['Otros'] - pron['SSFV'], np.where((pron['Otros'] - pron['SSFV'])>=0, pron['Otros'],0)),name="Otros",marker_color="gray"))
        #fig.add_trace(go.Bar(x=df["ds"], y = pron['Aires Acondicionados'] - pron['SSFV'],name="AA",marker_color="lightblue"))

        #fig.add_trace(go.Bar(x=df["ds"],y=np.where(df['value'] >= 0, pron['SSFV'], - pron["SSFV"] + pron['Otros'] + pron["Aires Acondicionados"]),name="Solar",marker_color="orange"))
        #fig.add_trace(go.Bar(x=df["ds"],y=np.where(df['value'] >= 0, pron['Otros'] - pron['SSFV'], 0), name="Otros",marker_color="gray"))
        #fig.add_trace(go.Bar(x=df["ds"],y=np.where(df['value'] >= 0, np.maximum(0,pron['Aires Acondicionados'] - pron['Otros'] + pron["SSFV"]), 0),name="Aires Acondicionados",marker_color="lightblue"))
        fig.add_trace(go.Scatter(x=df["ds"], y=round(df["value"],1), mode="lines",name='General',line=dict(color='black')))

    else:
        fig.add_trace(go.Scatter(x=df["ds"], y=df["value"], mode="lines",name='Real'))
        fig.add_trace(go.Scatter(x=df["ds"], y=pron, mode="lines",name='Pronosticado'))
    fig.update_layout(title="", margin=dict(t=30, b=0), barmode="relative",
                      xaxis=dict(domain=[0.1, 0.99],title="Fecha", showline=True, linecolor='black', showgrid=False, zeroline=False),
                      yaxis_title="Consumo (kWh)",legend=dict(orientation="h", x=0.5, xanchor="center", y=1.4, yanchor="top"), height=200)
    st.plotly_chart(fig, use_container_width=True)

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

def display_submedidores(submedidores, nombres_submedidores, icons, metrics, db, pron):
    cols = st.columns(len(submedidores))
    for i, label in enumerate(submedidores):
        nombre = nombres_submedidores.get(label, label)
        with cols[i]:
            with st.container(border=True):
                ca,cb = st.columns([1, 2], vertical_alignment='center')
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

def mostrar_imagen(path, width=150):
    img64 = base64.b64encode(open(path, "rb").read()).decode()
    st.markdown(
        f'<div style="text-align:center;"><img src="data:image/png;base64,{img64}" width="{width}"></div>',
        unsafe_allow_html=True
    )

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
        col1, col2 = st.columns(2)
        with col1:
            st.metric("👥 Personas", f"{df_pers['value'].iloc[-1]:.0f}")
            st.markdown('<br>',unsafe_allow_html=True)
        with col2:
            st.metric("⚡Consumo", f"{round(df['value'].iloc[-1],1)} kWh", delta=f"{round(diferencia,1)} kWh", delta_color="inverse")

    return promedios.iloc[-1]

def get_icons():
    return {
        "General": "images/MedidorGen.png",
        "AC": "images/MedidorAA.png",
        "SSFV": "images/MedidorPV.png",
        "Otros": "images/MedidorOtros.png"
    }

def graficar_intensidad_heatmap(ruta_excel):
    df = pd.read_excel(ruta_excel)
    df["dia_semana"] = pd.Categorical(df["dia_semana"], categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], ordered=True)
    tabla = df.pivot(index='dia_semana', columns='hora', values='intensidad')
    custom_text = [[f"Día: {dia}<br>Hora: {hora}<br>Intensidad: {tabla.loc[dia, hora]:.1f}" for hora in tabla.columns] for dia in tabla.index]
    fig = go.Figure(go.Heatmap(z=tabla.values, x=tabla.columns, y=tabla.index,colorscale='Viridis', colorbar_title='Intensidad', text=custom_text, hoverinfo='text', xgap=1, ygap=1))
    fig.update_layout(title='Programación de Intensidad Habitual de Aires', xaxis=dict(title='Hora', showgrid=True), yaxis=dict(title='Día', showgrid=True), template='simple_white')
    st.plotly_chart(fig, use_container_width=True)

def display_smart_control(db1, db2, t_int):
    personas = db1.loc[db1["unique_id"] == 'Ocupacion'].iloc[-1, 2]
    t_ext = db2['T2M'].iloc[-1]
    df_temp = db1[db1["unit"] == '°C']
    zonas = [f"T{i}" for i in range(1, 11)]
    Z = [df_temp[df_temp["unique_id"].isin(zonas[i:i+2])][['value']].mean().iloc[-1] for i in range(0, 10, 2)]

    with st.container(border=True):
        st.markdown("#### Control del Edificio")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            col.metric(f"🌡️ Temp. Zona {i+1}", f"{Z[i]:.1f} °C")
        with st.container(key="styled_tabs_2"):
            tab1, tab2 = st.tabs(["Programación Estándar", "Programación IA"])
            with tab1.container(key='cont-BMS'):
                graficar_intensidad_heatmap("BMS/programacion_bms.xlsx")

            with tab2.container(key='cont-BMS-IA'):
                ruta = 'BMS/programacion_bms.xlsx'
                resultado, pronostico, base, vel = tools.agenda_bms(ruta, datetime.now() - pd.Timedelta(hours=5), personas, t_ext, t_int)
                st.info(resultado)
                unidades = tools.seleccionar_unidades(pronostico, base)
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    col.markdown(f"<div style='text-align: center;'>Aire {i+1}</div>", unsafe_allow_html=True)
                    estado = ["🔴 Apagado", "🟢 Encendido", "🤖 Auto"][unidades[i]]
                    estilo = [col.error, col.success, col.warning][unidades[i]]
                    estilo(estado)
                    col.progress(vel / 7 if unidades[i] == 1 else 0)

                with open("./images/Piso-1-lado-A-aires-Encendidos.gif", "rb") as f:
                    data_url = base64.b64encode(f.read()).decode("utf-8")
                st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="aire gif">', unsafe_allow_html=True)

                enlaces = [("Piso 1: Lado A", ""), ("Piso 1: Lado B", "/Piso_1_Lado_B"), ("Piso 2: Lado A", "/Piso_2")]
                for col, (nombre, ruta) in zip(st.columns(3), enlaces):
                    col.markdown(f'<div style="text-align: center;"><a href="http://192.168.5.200:3000{ruta}" target="_blank">{nombre}</a></div>', unsafe_allow_html=True)
