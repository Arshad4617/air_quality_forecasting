import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima_model import ARIMA
import pickle


df = pd.read_csv('Dataset.csv',index_col='Year', parse_dates=True)
#FEATURE ENGINEERING
df['sqrt_trans'] = np.sqrt(df.CO2)

#MODEL
ARIMAmodel = ARIMA(df['sqrt_trans'], order=(1,0,1)) #notice p,d and q value here
ARIMA_model_fit = ARIMAmodel.fit(disp = 0)

# Saving model to disk
pickle.dump(ARIMA_model_fit, open('model.pkl','wb'))