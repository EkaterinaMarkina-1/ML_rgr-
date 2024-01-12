import streamlit as st
import pandas as pd
import pickle
from category_encoders.binary import BinaryEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow import keras

df_reg= pd.read_csv('mumbai_houses_task.csv')

st.header("Получение прогноза для конкретного экземпляра")

Prise = st.slider('Choose prise', min_value=0,
                  max_value=1000000000000000, value=(20, 50))
Area = st.slider('Choose area', min_value=0,
                 max_value=10000000, value=(20, 50))
Latitude = st.slider('Choose latitude', min_value=0,
                     max_value=1000000000000000, value=(20, 50))
Longitude = st.slider('Choose longitude', min_value=0,
                      max_value=1000000000000000, value=(20, 50))

Bedrooms = st.multiselect('Choose bedrooms', df_reg['bedrooms'].unique())
Bathrooms = st.multiselect('Choose bathrooms', df_reg['bathrooms'].unique())
Balconyv = st.multiselect('Choose balconyv', df_reg['balconyv'].unique())
Status = st.multiselect('Choose status', df_reg['status'].unique())
Neworold = st.multiselect('Choose neworold', df_reg['neworold'].unique())
Parking = st.multiselect('Choose parking', df_reg['parking'].unique())
Furnished_status = st.multiselect(
    'Choose furnished_status', df_reg['furnished_status'].unique())
Lift = st.multiselect('Choose lift', df_reg['lift'].unique())
Type_of_building = st.multiselect(
    'Choose type_of_building', df_reg['type_of_building'].unique())

feature_0 = st.toggle("feature_0")
feature_1 = st.toggle("feature_1")
feature_2 = st.toggle("feature_2")
feature_3 = st.toggle("feature_3")
feature_4 = st.toggle("feature_4")
feature_5 = st.toggle("feature_5")
feature_6 = st.toggle("feature_6")
feature_7 = st.toggle("feature_7")
feature_8 = st.toggle("feature_8")
feature_9 = st.toggle("feature_9")

# Create the new DataFrame
new_data = [[Prise, Area, Latitude, Longitude, Bedrooms, Bathrooms, Balconyv, Status, Neworold, Parking, Furnished_status, Lift, Type_of_building, feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9]]
new_columns = ['Prise', 'Area', 'Latitude', 'Longitude', 'Bedrooms', 'Bathrooms', 'Balconyv', 'Status', 'Neworold', 'Parking', 'Furnished_status', 'Lift', 'Type_of_building', 'feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9']
new_df = pd.DataFrame(data=new_data, columns=new_columns)

st.subheader("Введённые данные")
st.write(new_df)


predictions = []
btn = st.button("Рассчитать")
if btn:
    with open(r"1_multi_dimensional_regression.pickle", "wb") as f:
        reg = pickle.load(f)
        pred = reg.predict(df_reg)[0]
        predictions.append(pred)
        st.write("Множественная линейная регрессия: " + str(round(pred)))

    with open(r"3_gradient_boosting_regression.pickle", "rb") as f:
        gbr = pickle.load(f)
        pred = gbr.predict(df_reg)[0]
        predictions.append(pred)
        st.write("Градиентный бустинг: " + str(round(pred)))

    with open(r"4_bagging_regression.pickle", "rb") as f:
        br = pickle.load(f)
        pred = br.predict(df_reg)[0]
        predictions.append(pred)
        st.write("Бэггинг: " + str(round(pred)))

    with open(r"5_stacking_regression.pickle", "rb") as f:
        sr = pickle.load(f)
        pred = sr.predict(df_reg)[0]
        predictions.append(pred)
        st.write("Стекинг: " + str(round(pred)))

    nn = keras.models.load_model(r"6_neural_network_regression.h5")
    pred = nn.predict(df_reg)
    st.write("Полносвязная нейронная сеть:", str(round(int(pred[0]))))

    st.write("Усреднённое значение прогнозов:", str(round(np.array(predictions).mean())))
