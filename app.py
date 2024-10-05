import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



# Load the Iris dataset
def load_data():
    data=pd.read_csv('train.csv')
    return data
df=load_data()

# Streamlit app layout
st.title("Aplikasi visualisasi pemesanan hotel")
st.write("""
### Ini adalah apliasi streamlit yang menggunakan dataset pemesanan yang sudah clean
""")

st.write("Data yang sudah dibersihkan:")
st.write(df.head())

# Sidebar for user input
st.sidebar.header("User Input Parameters")
def user_input_features():
    lead_time=st.sidebar.slider("Load Time",0,365,100)
    stay_in_weeken_night=st.sidebar.slider("Stays in weekend Night",0,10,2)
    stays_in_week_nights=st.sidebar.slider("Stay in week night",0,15,5)
    adult=st.sidebar.slider("Adult",1,4,2)
    children=st.sidebar.slider("Children",0,5,1)
    babies=st.sidebar.slider("Babies",0,3,0)
    previous_cancellations=st.sidebar.slider("Previous Cancellations",0,5,0)
    adr=st.sidebar.slider("Average Daily Rate",0.0,500.0,100.0)

    data={
        'lead_time':lead_time,
        'stays_in_weekend_night':stay_in_weeken_night,
        'stays_in_week_nights':stays_in_week_nights,
        'adult':adult,
        'children':children,
        'babies':babies,
        'previous_cancellastion':previous_cancellations,
        'adr':adr
    }
    return pd.DataFrame(data,index=[0])
input_df=user_input_features()

#show input parameters
st.subheader("User Input Parameter")
st.write(input_df)

#Visualisasi
#ADR by hotel type
st.subheader("Rata-rata Tarif Harian(ADR) berdasarkan Jenis Hotel")
fig=px.bar(df,x='hotel',y='adr',color='hotel',title='Rata-rata tarif harian')
st.plotly_chart(fig)
#scater plot lead time vs ADR
st.subheader("Lead Time vs Tarif Harian")
fig2=px.scatter(df,x='lead_time',y='adr',color='hotel',title='Lead Time vs Tarif Harian')
st.plotly_chart(fig2)

#Prepare data for Machine Learning
df_ml=df[['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'previous_cancellations', 'adr', 'is_canceled']]
x=df_ml.drop('is_canceled',axis=1)
y=df_ml['is_canceked']

#split data into training and testing sets
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3,random_state=42)

model=Ran

