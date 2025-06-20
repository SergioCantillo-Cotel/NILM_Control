import streamlit as st
from utils import tools, viz

tools.quarter_autorefresh(key='cont')
credentials = tools.bigquery_auth()
db_pow, db_temp, db_occup = tools.read_bq_db(credentials)
lat, lon = 3.4793949016367822, -76.52284557701176
datos = tools.get_climate_data(lat, lon)
t_prom = tools.get_temp_prom(db_temp)
viz.display_smart_control_gen(db_occup,datos,t_prom,db_temp,db_pow)
