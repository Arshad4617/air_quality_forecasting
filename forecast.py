import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima_model import ARIMA
import pickle


st.set_page_config(
     page_title="Air_Quality_Forecasting",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
 )

#Loading Pickle file
model = pickle.load(open('model.pkl', 'rb'))

#header

st.title('Air Quality Forecasting')

col1, col2 = st.columns([2, 4])



#miking arrangesments for prediction

def predict():
    '''
    For rendering results on HTML GUI
    '''

    date = []
    for i in range(2015,2015+years_input):
        date.append('{}-01-01'.format(i))
    
    prediction = model.predict(start = date[0],end = date[-1])
    prediction = prediction * prediction
    pred_df = pd.DataFrame(prediction,columns = ['Prediction'])
    co2_data = pd.read_csv('Dataset.csv',index_col='Year', parse_dates=True)
    dd = pd.concat([co2_data.CO2,pred_df.Prediction],axis = 1)
    col1.subheader("Emission of CO2 for next {} Years".format(years_input))

    return col1.dataframe(pred_df),col2.line_chart(dd)  # Same as st.write(df)




co2_data = pd.read_csv('Dataset.csv',index_col='Year', parse_dates=True)


years_input = col1.slider('How Manay Years of prediction do you want', min_value=1, max_value=10)
if col1.button('forecast'):
    result = predict()
else :
    col2.line_chart(co2_data)






