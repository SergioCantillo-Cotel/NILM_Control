import streamlit as st
import base64
from datetime import datetime, timedelta
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
        fig.add_trace(go.Bar(x=df["ds"],y=np.where(df['value'] > 0, pron['Otros'], 0), name="Otros",marker_color="gray"))
        fig.add_trace(go.Bar(x=df["ds"],y=np.where(df['value'] > 0, - pron['SSFV'] + pron['Aires Acondicionados'], 0),name="Aires Acondicionados",marker_color="lightblue"))
        fig.add_trace(go.Scatter(x=df["ds"], y=round(df["value"],1), mode="lines",name='General',line=dict(color='black')))
    else:
        fig.add_trace(go.Scatter(x=df["ds"], y=df["value"], mode="lines",name='Real'))
        fig.add_trace(go.Scatter(x=df["ds"], y=pron, mode="lines",name='Pronosticado'))
    fig.update_layout(title="", margin=dict(t=30, b=0), barmode="relative",font=dict(family="Poppins", color="black"),
                      xaxis=dict(domain=[0.05, 0.95], title="Fecha", showline=True, linecolor='black', showgrid=False, zeroline=False, title_font=dict(color='black'),tickfont=dict(color='black')),
                      yaxis=dict(title="Consumo (kWh)", title_font=dict(color='black'), tickfont=dict(color='black')),
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=1.4, yanchor="top", font=dict(color="black")), height=210)
    st.plotly_chart(fig, use_container_width=True)

def display_submedidores(submedidores, nombres_submedidores, icons, metrics, db, pron):
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
                    if float(metrics[label]['energia']) < 0:
                        render_custom_metric(cb, nombre, f"{metrics[label]['energia']} kWh",f"{porc:.1f}%",'green')
                    elif float(metrics[label]['energia']) > 0:
                        render_custom_metric(cb, nombre, f"{metrics[label]['energia']} kWh",f"{porc:.1f}%",'red')
                    else:
                        render_custom_metric(cb, nombre, f"{metrics[label]['energia']} kWh",f"{porc:.1f}%")
                        
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
            graficar_consumo(df,pron,False)

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
            if diferencia > 0:
                render_custom_metric(col2, "⚡Consumo", f"{round(df['value'].iloc[-1],1)} kWh")
            elif diferencia < 0:
                render_custom_metric(col2, "⚡Consumo", f"{round(df['value'].iloc[-1],1)} kWh")
            else:
                render_custom_metric(col2, "⚡Consumo", f"{round(df['value'].iloc[-1],1)} kWh")

def get_icons():
    return {
        "General": "images/MedidorGen.png",
        "AC": "images/MedidorAA.png",
        "SSFV": "images/MedidorPV.png",
        "Otros": "images/MedidorOtros.png"
    }

def display_BMS_schedule(ruta_excel):
    df = pd.read_excel(ruta_excel)
    df["dia_semana"] = df["dia_semana"].map({
        "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
        "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"
    }).astype(pd.CategoricalDtype(
        categories=["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"], ordered=True
    ))
    tabla = df.pivot(index='dia_semana', columns='hora', values='intensidad')
    custom_text = [[f"Día: {dia}<br>Hora: {hora}:00<br>Intensidad: {tabla.loc[dia, hora]:.1f}%"
                    for hora in tabla.columns] for dia in tabla.index]

    fig = go.Figure(go.Heatmap(z=tabla.values,x=tabla.columns,y=tabla.index,colorscale='rdbu',
                               colorbar=dict(title='Intensidad',title_font=dict(family='Poppins', color='black'),
                                             tickfont=dict(family='Poppins', color='black')),
                                             text=custom_text,hoverinfo='text',xgap=1,ygap=1))

    fig.update_layout(
        margin=dict(t=10, b=0, l=0, r=0), template='simple_white', height=450, font=dict(family='Poppins', color='black'), 
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

def display_comparativa(db_AA,db_pers,db_t_ext=None,db_t_int=None):
    ruta = 'BMS/programacion_bms.xlsx'
    fig = go.Figure()
    now = pd.Timestamp.now().floor('15min') - pd.Timedelta(hours=5)
    inicio = now - pd.Timedelta(weeks=1)
    sch_BMS = (pd.DataFrame({'ds': pd.date_range(inicio, now, freq='15min')})
               .assign(dia_semana=lambda x: x.ds.dt.day_name(),hora=lambda x: x.ds.dt.hour)
               .merge(pd.read_excel('BMS/programacion_bms.xlsx').drop('promedio', axis=1),
                      on=['dia_semana', 'hora'],how='left'))
    
    db_AA['ds'] = db_AA['ds'].dt.floor('15min')[(db_AA['ds'] >= inicio) & (db_AA['ds'] <= now)]
    sch_RT = db_AA.groupby('ds')['value'].agg(lambda x: x.sum() * 100 / x.count()).reset_index()
    
    db_pers['ds'] = db_pers['ds'].dt.floor('15min')[(db_pers['ds'] >= inicio) & (db_pers['ds'] <= now)]
    pz = db_pers.copy()
    db_pers = db_pers.groupby('ds')['value'].sum().reset_index()
    
    db_t_ext['ds'] = db_t_ext['ds'].dt.floor('15min')[(db_t_ext['ds'] >= inicio) & (db_t_ext['ds'] <= now)]
    db_t_ext = db_t_ext.groupby('ds')['T2M'].sum().reset_index() 
    db_t_int = db_t_int[(db_t_int['ds'] >= inicio) &(db_t_int['ds'] <= now) &(db_t_int['unique_id'].str.match(r'^T(10|[1-9])$'))].copy()
    db_t_int['ds'] = db_t_int['ds'].dt.floor('15min')
    db_t_int = db_t_int.pivot_table(index='ds', columns='unique_id', values='value', aggfunc='mean').mean(axis=1).reset_index(name='promedio_T')
    sch_IA = []
    fechas_unicas = sorted(db_pers['ds'].drop_duplicates())
    for i in range(len(db_pers)):
        _, pronostico = tools.agenda_bms(ruta,db_pers['ds'].iloc[i],db_pers['value'].values[i],db_t_ext['T2M'].values[i],db_t_int['promedio_T'].values[i])
        carga = tools.nueva_carga(pronostico, pz[(pz['ds'] == fechas_unicas[i]) & (pz['unique_id'] != 'Flotantes')])
        sch_IA.append({'ds': db_pers['ds'].iloc[i], 'intensidad_IA': carga})
    sch_IA = pd.DataFrame(sch_IA)
    fig.add_trace(go.Scatter(x=sch_BMS["ds"], y=sch_BMS["intensidad"], mode="lines", name='Prog. BMS', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=sch_RT["ds"], y=sch_RT["value"], mode="lines", name='Comportamiento Real', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=sch_IA["ds"], y=sch_IA["intensidad_IA"], mode="lines", name='IA', line=dict(color='green')))    
    fig.update_layout(title="", margin=dict(t=30, b=0, l=20, r=20), font=dict(family="Poppins", color="black"),
                      xaxis=dict(domain=[0.05, 0.99], title="Fecha", showline=True, linecolor='black', showgrid=False, 
                                 zeroline=False, tickfont=dict(color='black'), title_font=dict(color='black')),
                      yaxis=dict(title="Capacidad de Refrigeración (%)", title_font=dict(color='black'), tickfont=dict(color='black')),
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1, yanchor="top"), height=510)
    st.plotly_chart(fig, use_container_width=True)
    dif_BMS_RT = pd.Series(np.where(sch_BMS['intensidad'] != 0, 100 * (sch_BMS['intensidad'] - sch_RT['value']) / sch_BMS['intensidad'], np.nan)).fillna(0)
    dif_BMS_IA = pd.Series(np.where(sch_BMS['intensidad'] != 0, 100 * (sch_BMS['intensidad'] - sch_IA['intensidad_IA']) / sch_BMS['intensidad'], np.nan)).fillna(0)
    return dif_BMS_RT.mean(), dif_BMS_IA.mean(), sch_IA

def display_temp_zonal(db1,db2):
    df_temp = db1[db1["unit"] == '°C']
    df_ocup = db2[(db2["ds"] == db2["ds"].max()) & (db2["unique_id"] != 'ocupacion_flotante')]
    zonas = [f"T{i}" for i in range(1, 11)]
    espacios = np.array([6, 21, 26, 30, 15])
    Z = [df_temp[df_temp["unique_id"].isin(zonas[i:i+2])][['value']].mean().iloc[-1] for i in range(0, 10, 2)]
    with st.container(border=False, key="temp-zon"):
        st.markdown("#### 📍Zonas Monitoreadas")
        zonas = ['Sala de Juntas <br>Cubículos <br>(P2)',' Gerencia General <br> TI <br> (P2)',
                 'Gestión Humana <br> Eficiencia Preventa <br>(P1)','Contabilidad <br> Sala de Juntas <br> (P1)','Gestión Humana <br> Depto. Jurídico <br> (P1)']
        cols = st.columns(5, vertical_alignment='bottom')
        for i, col in enumerate(cols):
            col.markdown(f"""<div class="custom-metric">{zonas[i]}<br><br><div class="value-mon">🌡️ {Z[i]:.1f} °C <br>👥 {df_ocup.value.iloc[i]:.0f}/{espacios[i]:.0f} </div></div>""", unsafe_allow_html=True)

def display_mgen(db,rango_ev,fecha_int,t_ext,ocup,intensidad,solar,t_int):
    med_Gen = db.loc[(db["unique_id"] == 'General') & 
                     (db["ds"] >= pd.Timestamp(rango_ev[0])) &
                     (db["ds"] <= pd.Timestamp(rango_ev[1])), ['ds', 'value']]
    
    t_ext = t_ext.loc[(t_ext['ds'] >= pd.Timestamp(fecha_int)) & (t_ext['ds'] <= pd.Timestamp(rango_ev[1]))].copy()
    
    ocup = ocup.groupby('ds')['value'].sum().reset_index().rename(columns={'value': 'Ocupacion'})
    ocup = ocup.loc[(ocup['ds'] >= pd.Timestamp(fecha_int)) & (ocup['ds'] <= pd.Timestamp(rango_ev[1]))].copy()
    
    intensidad = intensidad.loc[(intensidad['ds'] >= pd.Timestamp(fecha_int)) & (intensidad['ds'] <= pd.Timestamp(rango_ev[1]))].copy()
    
    solar = solar.loc[(solar['ds'] >= pd.Timestamp(fecha_int))&(solar['ds'] <= pd.Timestamp(rango_ev[1]))&(solar['unique_id'] == 'SSFV'),['ds', 'value']].copy()
    
    t_int = t_int.reset_index().assign(ds=lambda df: pd.to_datetime(df['ds']).dt.floor('15min'))  # Convierte 'ds' en columna
    t_int = t_int.loc[(t_int['ds'] >= pd.Timestamp(fecha_int)) & (t_int['ds'] <= pd.Timestamp(rango_ev[1]))].copy()
    
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
    
def display_smart_control_gen(db1, db2, t_int, db_AA=None):
    personas = db1[(db1["ds"] == db1["ds"].max())]['value'].sum()
    personas_zona = db1[(db1["ds"] == db1["ds"].max()) & (db1["unique_id"] != 'ocupacion_flotante')]
    t_ext = db2['T2M'].iloc[-1]
    ruta = 'BMS/programacion_bms.xlsx'
    zonas = ['Sala de Juntas <br>Cubículos <br>(P2)',
             'Gerencia<br> Área TI <br> (P2)',
             'G. Humana <br> EE - Preventa <br>(P1)',
             'Contabilidad <br> Sala de Juntas <br> (P1)',
             'G. Humana <br> Depto. Jurídico <br> (P1)']
    
    with st.container(key="styled_tabs_2"):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Programación Estándar", "Estado Tiempo Real", "Programación IA", "Comparativa","Evaluación de Impacto"])

        with tab1.container(key='cont-BMS'):
            with st.container(key='SBC-BMS'):
                st.markdown("##### ❄️ Programación de Carga de refrigeración (BMS)")
                display_BMS_schedule(ruta)

        with tab2.container(key='cont-estAA'):
            estados_AA = db_AA[(db_AA['ds'] == db_AA['ds'].max()) & (db_AA['unique_id'].str.contains(r'Valvula_[13579]$', na=False))].sort_values(by='unique_id').copy()
            with st.container(key='SBC-RT'):
                cols_estAA = st.columns(5)
                for i, col in enumerate(cols_estAA):
                    cont_AA = col.container(border=True)
                    cont_AA.markdown(f"""<div class="custom-metric">{zonas[i]}<br></div>""", unsafe_allow_html=True)
                    estado = ["🔴 Apagado", "🟢 Encendido"][int(estados_AA.iloc[i,2])]
                    estilo = [cont_AA.error, cont_AA.success][int(estados_AA.iloc[i,2])]
                    estilo(estado)
                
        with tab3.container(key='cont-BMS-IA'):
            dia, pronostico = tools.agenda_bms(ruta, datetime.now()-pd.Timedelta(hours=5), personas, t_ext, t_int)
            unidades, vel, resultado, _ = tools.seleccionar_unidades(pronostico,personas_zona,datetime.now()-pd.Timedelta(hours=5),dia)
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
        
        with tab4.container(key='cont-comparativa'):
            estados_AA = db_AA[(db_AA['unique_id'].str.contains(r'Valvula_[13579]$', na=False))].sort_values(by='unique_id').copy()
            with st.container(key='SBC-graph-com'):
                col_A,col_B = st.columns([7, 3],vertical_alignment='center')
                with col_A:
                    dif_BMS_RT, dif_BMS_IA, int_IA = display_comparativa(estados_AA,db1,db2,db_AA)
                with col_B:
                    fig_BMS_IA = go.Figure(go.Indicator(mode="number",value=dif_BMS_IA, align = 'center',
                                                        number={'suffix': "%"}, domain={'x': [0, 1], 'y': [0, 1]},
                                                        title={'text': "Ahorro IA vs<br>Programación BMS", 'font': {'size': 14}}))
                    fig_BMS_IA.update_layout(margin=dict(t=80, b=20, l=20, r=20), height=190, font=dict(family="Poppins",color="black"))
                    col_B.plotly_chart(fig_BMS_IA, use_container_width=True, key='dif_BMS_IA',config={'displayModeBar': False})

                    fig_BMS_RT = go.Figure(go.Indicator(mode="number",value=dif_BMS_RT, align = 'center',
                                                        number={'suffix': "%"}, domain={'x': [0, 1], 'y': [0, 1]},
                                                        title={'text': "Ahorro Operacion RT vs<br>Programación BMS", 'font': {'size': 14}}))
                    fig_BMS_RT.update_layout(margin=dict(t=80, b=20, l=20, r=20), height=190, font=dict(family="Poppins",color="black"))
                    col_B.plotly_chart(fig_BMS_RT, use_container_width=True, key='dif_BMS_RT',config={'displayModeBar': False})

        with tab5.container(key='cont-impacto'):
            with st.container(key='SBC-impacto'):
                col_a, col_b = st.columns([1,1], vertical_alignment='center')
                with col_a:
                    rango_est = col_a.date_input("Periodo de evaluación", (pd.Timestamp.now() - pd.Timedelta(hours=5) - pd.Timedelta(days=7), pd.Timestamp.now() - pd.Timedelta(hours=5)), min_value='2025-06-11' ,key='periodo_estudio')
                with col_b:       
                    fecha_int = col_b.date_input("Fecha de Intervención", pd.Timestamp.now() - pd.Timedelta(hours=5) - pd.Timedelta(days=2), min_value=rango_est[0], max_value=rango_est[1] ,key='fecha_intervencion')
                display_mgen(db_Pow,rango_est,pd.Timestamp(fecha_int),db2[['ds','T2M']],db1[['ds','value']],int_IA,db_Pow,t_int)
