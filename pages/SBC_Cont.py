import streamlit as st
from utils import tools, viz

tools.quarter_autorefresh()
credentials = tools.bigquery_auth()
db_pow, db_temp, db_occup = tools.read_bq_db(credentials)
lat, lon = 3.4793949016367822, -76.52284557701176
datos = tools.get_climate_data_1m(lat, lon)
t_prom = tools.get_temp_prom(db_temp)
#opcion = st.sidebar.selectbox("Seleccione la zona:",["General", "(P2) Sala de Juntas - Cubículos","(P2) TI - Gerencia General","(P1) Gestión H. - Eficiencia E. - Preventa",
#                                            "(P1) Contabilidad - Recepción","(P1) Gestión H. - Depto Juridico"])

#if opcion == "General":
viz.display_smart_control_gen(db_occup,datos,t_prom,db_temp)
