import streamlit as st
import base64, holidays, gc
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from utils import tools

def toggle_visibility(key):
    st.session_state[key] = not st.session_state.get(key, True)

def graficar_consumo(df,pron,sub,fecha_ini=None):
    if fecha_ini is not None:
        fecha_ini_dt = pd.to_datetime(fecha_ini)
        df = df[df["ds"] >= fecha_ini_dt]
        if not sub:
            pron = pron[pron.index >= fecha_ini_dt]
        else:
            # Si pron es una Serie, asegurar que su índice es comparable con df["ds"]
            if hasattr(pron, "index") and hasattr(df, "ds"):
                # Alinear ambos por fecha
                mask = df["ds"] >= fecha_ini_dt
                df = df[mask]
                if isinstance(pron.index, pd.DatetimeIndex):
                    pron = pron[pron.index >= fecha_ini_dt]
                else:
                    # Si el índice no es fecha, simplemente recorta para igualar longitud
                    pron = pron[-len(df):]
            else:
                pron = pron
    fig = go.Figure()
    if not sub:
        solar = np.minimum(pron["SSFV"].values, df["value"].values)
        otros = np.minimum(pron["Otros"].values, np.maximum(df["value"].values - solar, 0))
        aires = np.maximum(df["value"].values - otros - solar, 0)  # Resta con aires, no con otros

        fig.add_trace(go.Bar(x=df["ds"], y=solar, name="Solar", marker_color="orange"))
        fig.add_trace(go.Bar(x=df["ds"], y=otros, name="Otros", marker_color="gray"))
        fig.add_trace(go.Bar(x=df["ds"], y=aires, name="Aires Acondicionados", marker_color="lightblue"))
        fig.add_trace(go.Scatter(x=df["ds"], y=np.round(df["value"],1), mode="lines", name='General', line=dict(color='black')))
    else:
        fig.add_trace(go.Scatter(x=df["ds"], y=df["value"], mode="lines",name='Real'))
        fig.add_trace(go.Scatter(x=df["ds"], y=pron, mode="lines",name='Pronosticado'))
    
    fig.update_layout(title="", margin=dict(t=30, b=0), barmode="relative",font=dict(family="Poppins", color="black"),
                      xaxis=dict(domain=[0.05, 0.95], title="Fecha", showline=True, linecolor='black', showgrid=False, zeroline=False, title_font=dict(color='black'),tickfont=dict(color='black')),
                      yaxis=dict(title="Consumo (kWh)", title_font=dict(color='black'), tickfont=dict(color='black')),
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=1.4, yanchor="top", font=dict(color="black")), height=210)
    st.plotly_chart(fig, use_container_width=True)

def display_submedidores(submedidores, nombres_submedidores, icons, metrics, db, pron, fecha_ini):
    cols = st.columns(len(submedidores))
    for i, label in enumerate(submedidores):
        nombre = nombres_submedidores.get(label, label)
        with cols[i]:
            with st.container(border=False, key=f'nilm-subm-{i}'):
                ca,cb = st.columns([1, 3], vertical_alignment='center')
                with ca:
                    mostrar_imagen(icons[label], 200)
                with cb:
                    porc = (pron.iloc[-1,i]/sum(pron.iloc[-1,:]))*100
                    color = 'green' if float(metrics[label]['energia']) < 0 else 'red' if float(metrics[label]['energia']) > 0 else '#6c757d'
                    render_custom_metric(cb, nombre, f"{metrics[label]['energia']} kWh", f"{porc:.1f}%", color)
                key_btn, key_vis = f"butt_{nombre}", f"vis_{nombre}"

                if st.button("Ver Detalle", key=key_btn,use_container_width=True):
                    toggle_visibility(key_vis)

                if st.session_state.get(key_vis, False):
                    df = db.loc[db["unique_id"] == nombre,['ds','value']]
                    graficar_consumo(df, pron[nombre], True, fecha_ini)
                    
def mostrar_imagen(path, width=150):
    with open(path, "rb") as f:
        img64 = base64.b64encode(f.read()).decode()
    st.markdown(f'<div style="text-align:center;"><img src="data:image/png;base64,{img64}" width="{width}"></div>', unsafe_allow_html=True)

def display_general(icons, metrics, db, pron, fecha_ini):
    with st.container(border=False, key='nilm-gen'):
        colg,colh = st.columns([1, 2], vertical_alignment='center')
        with colg:
            ca,cb = st.columns([1, 2], vertical_alignment='center')
            with ca:
                mostrar_imagen(icons['General'], 500)
            with cb:
                render_custom_metric(cb, "Medición General", metrics['General']['energia']+" kWh","100%")
        with colh:
            df = db.loc[db["unique_id"] == 'General',['ds','value']]
            graficar_consumo(df,pron,False,fecha_ini)

def display_extern_cond(datos, con_boton=False, lat=None, lon=None):
    with st.container(border=False, key='ext-cond'):
        st.markdown("#### 🌤️ Condiciones Externas")
        col1, col2, col3 = st.columns(3)
        with col1:
            render_custom_metric(col1, "🌡️ Temperatura", f"{datos['T2M'].iloc[-1]:.1f} °C")
            st.markdown('<br>',unsafe_allow_html=True)
        with col2:
            render_custom_metric(col2, "💧 Hum. Relativa", f"{datos['RH2M'].iloc[-1]:.1f} %")
        with col3:
            render_custom_metric(col3, "🌧️ Precipitaciones", f"{datos['PRECTOTCORR'].iloc[-1]:.1f} mm")

def display_intern_cond(db1,db2):
    df = db2.loc[db2["unique_id"] == 'General']
    df_pers = db1[(db1["ds"] == db1["ds"].max())]
    #df_pers = db1.loc[db1["unique_id"] == 'Ocupacion']
    if len(df) >= 2:
        diferencia = (df["value"].iloc[-1] - df["value"].iloc[-2])
    else:
        diferencia = None
    with st.container(border=False,key='int-cond'):
        st.markdown("#### 🏭 Condiciones Internas")
        col1, col2 = st.columns(2)
        with col1:
            render_custom_metric(col1, "👥 Personas", f"{df_pers['value'].sum():.0f}")
            st.markdown('<br>',unsafe_allow_html=True)
        with col2:
            render_custom_metric(col2, "⚡Consumo", f"{df['value'].iloc[-1]:.1f} kWh")

def get_icons():
    return {
        "General": "images/MedidorGen.png",
        "AC": "images/MedidorAA.png",
        "SSFV": "images/MedidorPV.png",
        "Otros": "images/MedidorOtros.png"
    }

def display_BMS_adj_sch(ruta_excel):
    df = pd.read_excel(ruta_excel, sheet_name="Raw")
    # Crear columna de hora:minuto para mayor resolución
    df["HORA_MIN"] = df["HORA"].astype(str).str.zfill(2) + ":" + df["MIN"].astype(str).str.zfill(2)
    df["DIA_SEMANA"] = df["DIA_SEMANA"].map({
        "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
        "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"
    }).astype(pd.CategoricalDtype(
        categories=["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"], ordered=True
    ))
    # Pivotear usando hora_min en vez de solo hora
    tabla = df.pivot(index='DIA_SEMANA', columns='HORA_MIN', values='INTENSIDAD')
    custom_text = [[f"Día: {dia}<br>Hora: {hora}<br>Intensidad: {tabla.loc[dia, hora]:.1f}%"
                    for hora in tabla.columns] for dia in tabla.index]

    fig = go.Figure(go.Heatmap(z=tabla.values, x=tabla.columns, y=tabla.index,colorscale='rdbu', 
        colorbar=dict(title='Intensidad',title_font=dict(family='Poppins', color='black'),tickfont=dict(family='Poppins', color='black')),
        text=custom_text, hoverinfo='text',xgap=1,ygap=1))

    fig.update_layout(
        margin=dict(t=10, b=0, l=0, r=0), template='simple_white',height=450, font=dict(family='Poppins', color='black'),
        xaxis=dict(title='Hora', showgrid=True, tickfont=dict(family='Poppins', color='black')),
        yaxis=dict(title='Día', showgrid=True, tickfont=dict(family='Poppins', color='black'))
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    return tabla

def render_custom_metric(col, label, value, delta=None,color='#6c757d',sym=""):
    html = f"""<div class="custom-metric"><div class="label">{label}</div><div class="value">{value}</div>"""
    if delta:
        delta = f"{sym+delta}"
        html += f"""<div class="delta" style="color:{color};">{delta}</div>"""
    html += "</div>"
    col.markdown(html, unsafe_allow_html=True)

@st.cache_data(show_spinner="Calculando comparativa...")
def calcular_comparativa_cached(db_AA, db_pers, db_t_ext, db_t_int):
    return tools.calcular_comparativa(db_AA, db_pers, db_t_ext, db_t_int)

def display_comparativa(db_AA, db_pers, db_t_ext=None, db_t_int=None):
    sch_BMS, sch_RT, sch_IA, dif_BMS_RT, dif_BMS_IA = calcular_comparativa_cached(db_AA, db_pers, db_t_ext, db_t_int)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sch_BMS["ds"], y=sch_BMS["INTENSIDAD"], mode="lines", name='Prog. BMS', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=sch_RT["ds"], y=sch_RT["value"], mode="lines", name='Comportamiento Real', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=sch_IA["ds"], y=sch_IA["intensidad_IA"], mode="lines", name='IA', line=dict(color='green')))    
    fig.update_layout(title="", margin=dict(t=30, b=0, l=20, r=20), font=dict(family="Poppins", color="black"),
                      xaxis=dict(domain=[0.05, 0.99], title="Fecha", showline=True, linecolor='black', showgrid=False, 
                                 zeroline=False, tickfont=dict(color='black'), title_font=dict(color='black')),
                      yaxis=dict(title="Capacidad de Refrigeración (%)", title_font=dict(color='black'), tickfont=dict(color='black')),
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1, yanchor="top"), height=510)
    st.plotly_chart(fig, use_container_width=True)
    return dif_BMS_RT, dif_BMS_IA, sch_IA

def display_temp_zonal(db1, db2):
    df_temp = db1[db1["unit"] == '°C']
    df_AA = db1[(db1['ds'] == db1['ds'].max()) & (db1['unique_id'].str.contains(r'Valvula_[13579]$', na=False))].sort_values(by='unique_id').copy()
    df_ocup = db2[(db2["ds"] == db2["ds"].max()) & (db2["unique_id"] != 'ocupacion_flotante')]
    zonas = [f"T{i}" for i in range(1, 11)]
    espacios = np.array([6, 21, 26, 30, 15], dtype='int8')
    Z = [df_temp[df_temp["unique_id"].isin(zonas[i:i+2])]['value'].mean() for i in range(0, 10, 2)]
    with st.container(border=False, key="temp-zon"):
        st.markdown("#### 📍Zonas Monitoreadas")
        zonas_nombres = [
            'Sala de Juntas <br>Cubículos <br>(P2)', ' Gerencia <br> Depto. TI <br> (P2)',
            'G. Humana <br> EE - Preventa <br>(P1)', 'Contabilidad <br> Sala de Juntas <br> (P1)',
            'Nómina <br> Depto. Jurídico <br> (P1)'
        ]
        cols = st.columns(5, vertical_alignment='bottom')
        for i, col in enumerate(cols):
            cont_AA = col.container(border=True)
            estado = ["OFF 🔴", "ON 🟢"][int(df_AA.iloc[i, 2])]
            color_estado = ["#dc3545", "#28a745"][int(df_AA.iloc[i, 2])]  # rojo / verde

            cont_AA.markdown(
                f"""<div class="custom-metric"> {zonas_nombres[i]}<br><br>
                <div class="value-mon">🌡️ {Z[i]:.1f} °C <br>👥 {df_ocup.value.iloc[i]:.0f}/{espacios[i]:.0f} <br> <div style="margin-top: 0.5rem; color: {color_estado}; margin-left: 0.1rem;">❄️ {estado}</div></div>
                </div>""",
                unsafe_allow_html=True
            )
    del df_temp, df_ocup, df_AA
    gc.collect()

def display_mgen(db, rango_ev, fecha_int, t_ext, ocup, intensidad, solar, t_int):
    med_Gen = db[(db["unique_id"] == 'General') & (db["ds"] >= pd.Timestamp(rango_ev[0])) & (db["ds"] <= pd.Timestamp(rango_ev[1]))][['ds', 'value']]
    t_ext = t_ext[(t_ext['ds'] >= pd.Timestamp(fecha_int)) & (t_ext['ds'] <= pd.Timestamp(rango_ev[1]))].copy()
    ocup = ocup.groupby('ds')['value'].sum().reset_index().rename(columns={'value': 'Ocupacion'})
    ocup = ocup[(ocup['ds'] >= pd.Timestamp(fecha_int)) & (ocup['ds'] <= pd.Timestamp(rango_ev[1]))].copy()
    intensidad = intensidad[(intensidad['ds'] >= pd.Timestamp(fecha_int)) & (intensidad['ds'] <= pd.Timestamp(rango_ev[1]))].copy()
    solar = solar[(solar['ds'] >= pd.Timestamp(fecha_int)) & (solar['ds'] <= pd.Timestamp(rango_ev[1])) & (solar['unique_id'] == 'SSFV')][['ds', 'value']].copy()
    t_int = t_int.reset_index().assign(ds=lambda df: pd.to_datetime(df['ds']).dt.floor('15min'))
    t_int = t_int[(t_int['ds'] >= pd.Timestamp(fecha_int)) & (t_int['ds'] <= pd.Timestamp(rango_ev[1]))].copy()
    for df in [solar, ocup, t_ext, t_int, intensidad]:
        df['ds'] = pd.to_datetime(df['ds']).dt.floor('15min')
    entradas_DT = solar.merge(ocup, on='ds').merge(t_ext, on='ds').merge(t_int, on='ds').merge(intensidad, on='ds')
    DT = tools.digital_twin(entradas_DT)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=med_Gen["ds"], y=round(med_Gen["value"],1), mode="lines",name='General',line=dict(color='black')))
    fig.add_trace(go.Scatter(x=DT["ds"], y=round(DT["Dig_Twin"],1), mode="lines",name='Digital Twin',line=dict(color='red')))
    fig.add_vline(x=fecha_int, line_width=2, line_dash="dash", line_color="red")
    fig.update_layout(title="", margin=dict(t=30, b=0),font=dict(family="Poppins", color="black"),
                      xaxis=dict(domain=[0.05, 0.95], title="Fecha", showline=True, linecolor='black', showgrid=False, zeroline=False, title_font=dict(color='black'),tickfont=dict(color='black')),
                      yaxis=dict(title="Consumo (kWh)", title_font=dict(color='black'), tickfont=dict(color='black')),
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=1.4, yanchor="top", font=dict(color="black")), height=450)
    st.plotly_chart(fig, use_container_width=True)
    del med_Gen, t_ext, ocup, intensidad, solar, t_int, entradas_DT, DT
    gc.collect()
    
def display_smart_control_gen(db1, db2, t_int, db_AA=None, db_Pow=None):
    personas = db1[(db1["ds"] == db1["ds"].max())]['value'].sum()
    personas_zona = db1[(db1["ds"].notna()) & (db1["ds"] == db1["ds"].max()) & (db1["unique_id"] != 'ocupacion_flotante')]
    t_ext = db2['T2M'].iloc[-1]
    ruta = 'BMS/programacion_bms.xlsx'
    ruta_2 = 'BMS/Prog_BMS.xlsx'
    zonas = ['Sala de Juntas <br>Cubículos <br>(P2)',
             'Gerencia<br> Área TI <br>(P2)',
             'G. Humana <br> EE - Preventa <br>(P1)',
             'Contabilidad <br> Sala de Juntas <br>(P1)',
             'Nómina <br> Depto. Jurídico <br>(P1)']
    
    with st.container(key="styled_tabs_2"):
        tab1, tab2, tab3, tab4 = st.tabs(["Programación Estándar", "Programación IA", "Comparativa", "Evaluación de Impacto"])

        with tab1.container(key='cont-BMS'):
            with st.container(key='SBC-BMS'):
                st.markdown("##### ❄️ Programación de Carga de refrigeración (BMS)")
                #display_BMS_schedule(ruta)
                display_BMS_adj_sch(ruta_2)
                
        with tab2.container(key='cont-BMS-IA'):
            dia, pronostico = tools.agenda_bms(ruta_2, pd.Timestamp.now(), personas, t_ext, t_int.iloc[-1])
            unidades, vel, resultado, _ = tools.seleccionar_unidades(pronostico,personas_zona,pd.Timestamp.now(),dia)
            st.info(resultado)
            with st.container(key='SBC-IA'):
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    cont_zonas = col.container(border=True)
                    cont_zonas.markdown(f"""<div class="custom-metric">{zonas[i]}</div>""", unsafe_allow_html=True)
                    estado = ["🔴 Apagado", "🟢 Encendido"][unidades[i]]
                    estilo = [cont_zonas.error, cont_zonas.success, cont_zonas.warning][unidades[i]]
                    estilo(estado)

                    fig1 = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=(vel[i] / 7) * 100 if unidades[i] == 1 else 0,
                        number={'suffix': "%"}, domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={'axis': {'range': [0, 100]},'bar': {'color': 'green', 'thickness': 1},}
                    ))
                    fig1.update_layout(margin=dict(t=0, b=0, l=20, r=30), height=120, font=dict(family="Poppins",color="black"))
                    cont_zonas.plotly_chart(fig1, use_container_width=True, key=f'vel{i}',config={'displayModeBar': False})
        
        with tab3.container(key='cont-comparativa'):
            estados_AA = db_AA[(db_AA['unique_id'].str.contains(r'Valvula_[13579]$', na=False))].sort_values(by='unique_id').copy()
            with st.container(key='SBC-graph-com'):
                col_A,col_B = st.columns([7, 2],vertical_alignment='center')
                with col_A:
                    dif_BMS_RT, dif_BMS_IA, int_IA = display_comparativa(estados_AA,db1,db2,db_AA)
                with col_B:
                    fig_BMS_IA = go.Figure(go.Indicator(mode="number",value=dif_BMS_IA, align = 'center',
                                                        number={'suffix': "%", 'valueformat': '.2f'}, domain={'x': [0, 1], 'y': [0, 1]},
                                                        title={'text': "Ahorro IA vs<br>Programación BMS", 'font': {'size': 14}}))
                    fig_BMS_IA.update_layout(margin=dict(t=80, b=20, l=20, r=20), height=190, font=dict(family="Poppins",color="black"))
                    col_B.plotly_chart(fig_BMS_IA, use_container_width=True, key='dif_BMS_IA',config={'displayModeBar': False})

                    fig_BMS_RT = go.Figure(go.Indicator(mode="number",value=dif_BMS_RT, align = 'center',
                                                        number={'suffix': "%", 'valueformat': '.2f'}, domain={'x': [0, 1], 'y': [0, 1]},
                                                        title={'text': "Ahorro Operacion RT vs<br>Programación BMS", 'font': {'size': 14}}))
                    fig_BMS_RT.update_layout(margin=dict(t=80, b=20, l=20, r=20), height=190, font=dict(family="Poppins",color="black"))
                    col_B.plotly_chart(fig_BMS_RT, use_container_width=True, key='dif_BMS_RT',config={'displayModeBar': False})
        
        with tab4.container(key='cont-impacto'):
            with st.container(key='SBC-impacto'):
                col_a, col_b = st.columns([1,1], vertical_alignment='center')
                with col_a:
                    rango_est = col_a.date_input("Periodo de evaluación", (pd.Timestamp.now() - pd.Timedelta(days=7), pd.Timestamp.now()), min_value='2025-06-11' ,key='periodo_estudio')
                with col_b:       
                    fecha_int = col_b.date_input("Fecha de Intervención", pd.Timestamp.now() - pd.Timedelta(days=2), min_value=rango_est[0], max_value=rango_est[1] ,key='fecha_intervencion')
                display_mgen(db_Pow,rango_est,pd.Timestamp(fecha_int),db2[['ds','T2M']],db1[['ds','value']],int_IA,db_Pow,t_int)
