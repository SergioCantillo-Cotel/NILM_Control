import streamlit as st
from utils import tools, viz, ia_model
import pandas as pd
from datetime import datetime

tools.quarter_autorefresh()
credentials = tools.bigquery_auth()
db_pow, db_oth, _ = tools.read_bq_db(credentials)
lat, lon = 3.4793949016367822, -76.52284557701176
datos = tools.get_climate_data(lat, lon)

nombres_submedidores = {"AC": "Aires Acondicionados","SSFV": "SSFV","otros": "Otras Cargas"}

modelo_IA = ia_model.get_IA_model()
caracteristicas = ia_model.datos_Exog(db_pow, datos).drop(columns=['ds'])
car2 = caracteristicas.copy()
y_hat_raw = modelo_IA.predict(caracteristicas.values.reshape(-1, 1, caracteristicas.shape[1]))
Y_hat_rec = ia_model.reconcile(car2,y_hat_raw)
Y_hat_df2 = pd.DataFrame(Y_hat_rec, columns=['Aires Acondicionados','SSFV','Otros'])
Y_hat_df2.index = db_pow.loc[db_pow["unique_id"] == 'General', "ds"].reset_index(drop=True)
metrics = tools.get_metrics(db_pow.loc[db_pow["unique_id"] == 'General',"value"].iloc[-1],
                      Y_hat_df2['Aires Acondicionados'].iloc[-1],
                      Y_hat_df2['SSFV'].iloc[-1],
                      Y_hat_df2['Otros'].iloc[-1])

submedidores = tools.get_submedidores(metrics)

viz.display_general(viz.get_icons(), metrics, db_pow, Y_hat_df2)
viz.display_submedidores(submedidores, nombres_submedidores, viz.get_icons(), metrics, db_pow, Y_hat_df2)
