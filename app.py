#%%writefile app.py
 
import pickle
import numpy as np
import streamlit as st
import pandas as pd
from pandas.tseries.offsets import DateOffset
from pathlib import Path
import streamlit_authenticator as stauth


#------ USER AUTHENTICATION-----------

names = ["Mobius DA"]
usernames = ["Mobius_Data_Analytics"]

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

credentials = {"usernames":{}}
for un, name, pw in zip(usernames, names, hashed_passwords):
    user_dict = {"name":name,"password":pw}
    credentials["usernames"].update({un:user_dict})

authenticator = stauth.Authenticate(credentials,"CorrosionRisk","abc123",cookie_expiry_days=1)

hide_streamlit_style = """<style> #MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
   

name,authetication_status,username = authenticator.login("LOGIN","main")

if authetication_status == False:
    st.error("Username/Password is incorrect")
if authetication_status == None:
    st.warning("Please enter your Username and Password")
    
#------ IF USER AUTHENTICATION STATUS IS TRUE  -----------   
if authetication_status:

    data  = pd.read_csv("Model_data.csv")
    #data['Date'] = pd.date_range(start='1/1/2022', periods=len(data), freq='D')
    data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
    data.set_index("Date", drop=True, inplace=True)

    #loading the exogenous features

    selected_features = open('Exog_features.sav', 'rb') 
    Exogenous_features = pickle.load(selected_features)
     
    # loading the trained model
    saved_model = open('SARIMAX_Full_Model_Best_1.sav', 'rb') 
    SARIMAX_Model = pickle.load(saved_model)
     
    @st.cache
      
    # defining the function which will make the prediction using the data which the user inputs 
    def prediction(days):    
        days = int(days)
        predictions = SARIMAX_Model.forecast(steps=int(days),exog=data[Exogenous_features].iloc[-int(days):]) 
        
        future_dates=[data.index[-1]+ DateOffset(days=x)for x in range(1,days+1)]
        future_dates_df=pd.DataFrame(index=future_dates[0:],columns=data.columns)
        future_df=pd.concat([data,future_dates_df])
        future_df['Forecasted Corrosion Rate'] = SARIMAX_Model.predict(start = data.shape[0], end = future_df.shape[0]-1, dynamic= True,exog=data[Exogenous_features].iloc[-int(days):])  
        return predictions,future_df
          
      
    # this is the main function in which we define our webpage  
    def main(): 
        authenticator.logout("Logout",'sidebar')
        st.sidebar.image("""https://cdn-icons-png.flaticon.com/512/4882/4882559.png""")
        
        global days,data,Exogenous_features
        #st.markdown(html_temp, unsafe_allow_html = True) 
        title = '<p style="font-family:sans-serif; color:black;text-align:center; font-size: 45px;"><b>Corrosion Rate Prediction</b></p>'
        subtitle = '<p style="font-family:sans-serif; color:grey;text-align:center; font-size: 20px;"><b>Time Series Forecasting</b></p>'
        st.markdown(title, unsafe_allow_html = True) 
        #st.markdown(subtitle, unsafe_allow_html = True) 
        # following lines create boxes in which user can enter data required to make prediction 
        
        header = '<p style="font-family:sans-serif; color:black;text-align:center; font-size: 20px;">Forecast Corrosion rate for next (days)</p>'
        st.markdown(header,unsafe_allow_html = True)
        days = st.slider("",min_value=1,max_value=90)
        st.text('Selected: {}'.format(days))
        
        result = ""
          
        # when 'Predict' is clicked, make the prediction and store it 
        
        if st.button("PREDICT"): 
            predictions,future_df = prediction(days)
            predictions = pd.DataFrame(predictions)
            predictions.rename(columns={'':'Date','predicted_mean':'Corrosion rate'},inplace=True)
            predictions = round(predictions,3)
            st.markdown("Table: Predicted Corrosion Rates for requested future days")            
            st.write(predictions)
            predictions =predictions.to_csv(index=True).encode('utf-8')
            st.download_button(label='Download Prediction(csv)',data=predictions,mime='text/csv',file_name='Download.csv')    
            chart_data = future_df[['Corrosion rate','Forecasted Corrosion Rate']]
            
            st.markdown("Lineplot showing Actual vs Predicted Corrosion Rate")
            st.line_chart(chart_data)
            
    if __name__=='__main__': 
        main()
    
