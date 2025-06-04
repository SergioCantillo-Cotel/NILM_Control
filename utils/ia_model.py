from keras.models import load_model
import pandas as pd
import numpy as np

def get_IA_model():
    IA_model = load_model('IA/NILM_Model_best.keras')
    return IA_model

def datos_Exog(db, datos):
    fut = db[db["unique_id"] == 'General'][['ds', 'value']].rename(columns={'value': 'Energia_kWh_General'})
    fut['ds'] = pd.to_datetime(fut['ds'])
    fut['DOW'] = fut['ds'].dt.dayofweek + 1
    fut['Hour'] = fut['ds'].dt.hour
    fut = fut.merge(datos.drop(columns='PRECTOTCORR', errors='ignore'), on='ds', how='left')
    return fut[['ds','Energia_kWh_General','DOW','Hour','T2M','RH2M']].sort_values(['ds'])

def reconcile(exog, pron):
    r = np.copy(pron)
    d = round(exog['Energia_kWh_General'],1) - (r[:,0] - r[:,1] + r[:,2])
    for i in range(len(r)):
        dow, h, di = exog['DOW'][i], exog['Hour'][i], r[i]
        wknd, work, sun, dia = dow in (6,7), 8<=h<=16, 11<=h<=13, 6<=h<=18

        if not dia:
            di[1], di[2] = 0, di[2] + d[i]
        elif wknd:
            di[2] += d[i]
        elif work:
            if sun:
                adj = min(d[i], di[1])
                di[1] -= adj
                di[2] += d[i] - adj
            else:
                adj = min(d[i]*0.5, di[1])
                di[1] -= adj
                di[2] += d[i] - adj
        else:
            adj = min(d[i], di[1])
            di[1] -= adj
            di[2] += d[i] - adj

        if dia and not wknd and not work and not sun:
            t = di[1] + di[2]
            if t > 0:
                di[1] += d[i] * di[1]/t
                di[2] += d[i] * di[2]/t
            else:
                di[1] += d[i]*0.5
                di[2] += d[i]*0.5

        di[1] = max(di[1], 0)
        di[2] = max(di[2], 0)

    return r
