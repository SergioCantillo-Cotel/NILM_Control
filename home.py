import streamlit as st
from datetime import datetime
from utils import tools
import pandas as pd
import pytz

st.set_page_config(page_title="Eficiencia Energética + IA Cotel", layout="wide")
tools.load_custom_css()
tools.quarter_autorefresh()
st.logo("images/cotel-logotipo.png", size="Large")

NILM = st.Page("pages/NILM.py", title="Cotel - Sede la Flora", icon="🏛️",)
SBMon = st.Page("pages/SBC_Mon.py", title="Monitoreo", icon="📈")
SBCon = st.Page("pages/SBC_Cont.py", title="Climatización", icon="❄️")
pg = st.navigation({"⚡ Medición Inteligente no Intrusiva": [NILM],
                    "🏢 Smart Building Control": [SBMon,SBCon]})
pg.run()

zona = pytz.timezone("America/Bogota")
ahora = pd.Timestamp(datetime.now(zona)).floor('15min').strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"""<div class="footer">🔄 Esta página se actualiza cada 15 minutos. Última actualización: {ahora}</div>
""", unsafe_allow_html=True)
