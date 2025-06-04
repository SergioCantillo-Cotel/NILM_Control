import streamlit as st
from datetime import datetime
from utils import tools
import pandas as pd
import pytz

st.set_page_config(page_title="Home", layout="wide")
tools.load_custom_css()
st.logo("images/cotel-logotipo.png", size="Large")
NILM = st.Page("pages/Pagina1.py", title="Medición Inteligente no Intrusiva", icon="⚡",)
SBCon = st.Page("pages/Pagina2.py", title="Smart Building Control", icon="🏢")

pg = st.navigation([NILM, SBCon])
pg.run()

zona = pytz.timezone("America/Bogota")
ahora = pd.Timestamp(datetime.now(zona)).floor('15min').strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"🔄 Esta página se actualiza cada 15 minutos. Última actualización: {ahora}")
