import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data
@st.cache_data  # Cache data to speed up loading
def load_data():
    df = pd.read_csv('train.csv')  # Ganti dengan nama dataset Anda
    return df

# Preprocessing
def preprocess_data(df):
    df_visualization = df.copy()
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
    df.drop(['company'], axis='columns', inplace=True)
    df['children'].fillna(df['children'].median(), inplace=True)
    df['agent'].fillna(df['agent'].median(), inplace=True)
    df['country'].fillna(df['country'].mode()[0], inplace=True)

    columns_to_drop = ['reservation_status_date', 'bookingID', 'reservation_status', 'arrival_date_month', 'date']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    label_encoder = LabelEncoder()
    if 'season' in df.columns:
        df['season'] = label_encoder.fit_transform(df['season'])
    df['deposit_type'] = label_encoder.fit_transform(df['deposit_type'])
    df['customer_type'] = label_encoder.fit_transform(df['customer_type'])
    
    column_to_encode = ['hotel', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type']
    for col in column_to_encode:
        df[col] = df[col].astype(str)

    df_encoded = pd.get_dummies(df, columns=column_to_encode, drop_first=True, dtype=int)
    return df_encoded, df_visualization

# Train model
@st.cache_resource  # Cache the model for later use
def train_model(df_encoded):
    X = df_encoded.drop(columns=['is_canceled'])  # Features
    y = df_encoded['is_canceled'].astype(int)  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, scaler

# Load data
df = load_data()
df_encoded, df_visualization = preprocess_data(df)
model, scaler = train_model(df_encoded)

# Streamlit app
st.title("Hotel Reservation Cancellation Prediction")

# Tab untuk visualisasi dan fitur prediksi
tab1, tab2 = st.tabs(["Visualisasi", "Prediksi"])

# Tab Visualisasi
with tab1:
    st.subheader("Visualisasi Data")

    # Visualisasi 1: Distribusi lead_time berdasarkan status pembatalan
    fig1 = px.histogram(df_visualization, x='lead_time', color='is_canceled', title='Distribusi Lead Time Berdasarkan Status Pembatalan',
                         color_discrete_map={'0': 'green', '1': 'red'})  # Mengatur warna untuk status pembatalan
    st.plotly_chart(fig1)

    # Visualisasi 2: Segmen Pasar vs Pembatalan menggunakan Plotly
    # Menyiapkan data untuk stacked bar chart
    segmentation_cancellation = df_visualization.groupby(['market_segment', 'is_canceled']).size().reset_index(name='count')

    # Menghitung total untuk setiap market_segment
    total_counts = segmentation_cancellation.groupby('market_segment')['count'].sum().reset_index()

    # Menggabungkan total_counts dengan segmentation_cancellation untuk sorting
    segmentation_cancellation = segmentation_cancellation.merge(total_counts, on='market_segment', suffixes=('', '_total'))

    # Mengurutkan berdasarkan total_counts
    segmentation_cancellation = segmentation_cancellation.sort_values(by='count_total', ascending=False)

    # Menggunakan plotly untuk stacked bar chart
    fig2 = px.bar(segmentation_cancellation, 
                x='market_segment', 
                y='count', 
                color='is_canceled', 
                title='Segmen Pasar vs Pembatalan',
                color_discrete_map={'0': 'green', '1': 'red'},
                barmode='stack')

    st.plotly_chart(fig2)
# Tab Prediksi
with tab2:
    st.subheader("Fitur Prediksi")

    # Sidebar for user inputs
    def user_input_features():
        hotel = st.selectbox("Hotel", ("City Hotel", "Resort Hotel"))
        lead_time = st.slider("Lead Time", 0, 500, 30)
        arrival_date_week_number = st.slider("Arrival Date Week Number", 1, 53, 25)
        number_of_adults = st.slider("Number of Adults", 0, 30, 2)
        number_of_children = st.slider("Number of Children", 0, 30, 0)
        number_of_special_requests = st.slider("Number of Special Requests", 0, 5, 0)

        data = {
            'hotel': hotel,
            'lead_time': lead_time,
            'arrival_date_week_number': arrival_date_week_number,
            'number_of_adults': number_of_adults,
            'number_of_children': number_of_children,
            'number_of_special_requests': number_of_special_requests
        }

        return pd.DataFrame(data, index=[0])

    # Get user input
    input_data = user_input_features()

    # Transform the input data
    input_data_encoded = pd.get_dummies(input_data, drop_first=True)

    # Ensure input_data_encoded has the same columns as training data
    missing_cols = set(df_encoded.columns) - set(input_data_encoded.columns)
    for col in missing_cols:
        input_data_encoded[col] = 0  # Add missing columns with value 0

    input_data_encoded = input_data_encoded[df_encoded.columns.drop('is_canceled')]

    # Standardize input data
    input_data_scaled = scaler.transform(input_data_encoded)

    # Predicting
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    # Display results
    st.subheader('Prediction')
    st.write('Cancelled' if prediction[0] == 1 else 'Not Cancelled')

    st.subheader('Prediction Probability')
    st.write(prediction_proba)
