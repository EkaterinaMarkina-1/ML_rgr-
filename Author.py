import streamlit as st
import pandas as pd
import pickle
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as ts


df= pd.read_csv("mumbai_houses_task.csv")
with open(r"1_multi_dimensional_regression.pkl", "rb") as f:
            reg = pickle.load(f)
with open(r"3_gradient_boosting_regression.pkl", "rb") as f:
            gbr = pickle.load(f)
with open(r"4_bagging_regression.pkl", "rb") as f:
            br = pickle.load(f)
with open(r"5_stacking_regression.pkl", "rb") as f:
            sr = pickle.load(f)
nn = ts.keras.models.load_model(r"neuroReg.h5")

def main():
    select_page = st.sidebar.selectbox("Page list", ("Title","DataSet description", "Visualization","Predict"), key = "Select")
    if (select_page == "Title"):
        title_page()

    elif (select_page == "DataSet description"):
        description_page()

    elif (select_page == "Predict"):
        prediction_page()

    elif (select_page == "Visualization"):
        visualization()

def title_page():
    st.title("Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных")


    st.header("Автор")
    st.write("ФИО: Маркина Екатерина Константиновна")
    st.write("Группа: ФИТ-222")
    st.write("2023 год")

def description_page():
    st.title("Информация о наборе данных")
    st.write()
    st.header("Тематика датасета")
    st.write("Информация о стоимости жилья в Мумбаях")
    st.header("Описание признаков")
    st.write("- price: Цена недвижимости. Этот признак указывает на стоимость жилья или коммерческой недвижимости. ")
    st.write("- area: Площадь недвижимости. Этот признак обозначает общую площадь жилого помещения или коммерческого объекта.")
    st.write("- latitude и longitude: Географические координаты недвижимости. Эти признаки указывают на местоположение объекта недвижимости.")
    st.write("- bedrooms и bathrooms: Количество спален и ванных комнат. Эти признаки определяют количество спален и ванных комнат в жилом помещении.")
    st.write("- balcony: Наличие балкона. Этот признак указывает, есть ли балкон в жилом помещении.")
    st.write("- status: Статус недвижимости. Этот признак может обозначать, например, продается ли недвижимость или сдается в аренду.")
    st.write("- neworold: Старая или новая недвижимость. Этот признак указывает на то, является ли недвижимость новой или уже использованной.")
    st.write("- parking: Наличие парковки. Этот признак указывает на наличие парковочных мест для автомобилей.")
    st.write("- furnished_status: Статус меблировки. Этот признак указывает на то, предлагается ли недвижимость с мебелью или без нее.")
    st.write("- lift: Наличие лифта. Этот признак указывает на наличие лифта в здании.")
    st.write("- type_of_building: Тип здания. Этот признак определяет тип здания, такой как квартирный дом, частный дом, офисное здание и т. д.")



def prediction_page():
    st.header("Получение прогноза для конкретного экземпляра")

    Prise = st.slider('Choose prise',0,1000000000000000)
    Area = st.slider('Choose area', 0,10000000)
    Latitude = st.slider('Choose latitude', 0,1000000000000000)
    Longitude = st.slider('Choose longitude', 0,1000000000000000)

    df= pd.read_csv("mumbai_houses_task.csv")

    Bedrooms = st.number_input('Choose the number of bedrooms', min_value=1,max_value=99,value=1,step=1)
    Bathrooms = st.number_input('Choose the number of bedrooms', min_value=1,max_value=100,value=1,step=1)
    Balconyv = st.number_input('Choose the number of balconyv', min_value=1,max_value=101,value=1,step=1)
    Status = st.number_input('Choose status', min_value=1,max_value=102,value=1,step=1)
    Neworold = st.number_input('Choose the number of neworold', min_value=1,max_value=103,value=1,step=1)
    Parking = st.number_input('Choose the number of parking', min_value=1,max_value=104,value=1,step=1)
    Furnished_status = st.number_input('Choose  furnished_status', min_value=1,max_value=105,value=1,step=1)
    Lift = st.number_input('Choose the number of lifts', min_value=1,max_value=106,value=1,step=1)
    Type_of_building = st.number_input('Choose type_of_building', min_value=1,max_value=107,value=1,step=1)


    # Create the new DataFrame
    new_data = [[ Area, Latitude, Longitude, Bedrooms, Bathrooms, Balconyv, Status, Neworold, Parking, Furnished_status, Lift, Type_of_building]]
    new_columns = [ 'area', 'latitude', 'longitude', 'bedrooms', 'bathrooms', 'balconyv', 'status', 'neworold', 'parking', 'furnished_status', 'lift', 'type_of_building']
    new_df = pd.DataFrame(data=new_data, columns=new_columns)

    st.subheader("Введённые данные")
    st.write(new_df)


    predictions = []
    btn = st.button("Рассчитать")
    if btn:
        pred = reg.predict(new_df)[0]
        predictions.append(pred)
        st.write("Множественная линейная регрессия: " + str(round(pred)))

        pred = gbr.predict(new_df)[0]
        predictions.append(pred)
        st.write("Градиентный бустинг: " + str(round(pred)))

        
        pred = br.predict(new_df)[0]
        predictions.append(pred)
        st.write("Бэггинг: " + str(round(pred)))

        
        pred = sr.predict(new_df)[0]
        predictions.append(pred)
        st.write("Стекинг: " + str(round(pred)))

        
        pred = nn.predict(new_df)
        st.write("Полносвязная нейронная сеть:", str(round(int(pred[0]))))

    st.title("Предсказание датасета")

    file = st.file_uploader("Загрузите непредобработанный файл в формате .csv")

    if file:
        df = pd.read_csv(file)
        if 'price' in df.columns:
            df.pop('price')
        getPredButton1=st.button("Получить предсказание")
        if getPredButton1:
            linReg_result=reg.predict(df)
            st.write("Результат множественной линейной регрессии:",pd.DataFrame(linReg_result,columns=['predicted_price']))

            baggingReg_result=br.predict(df)
            st.write("Результат BaggingRegressor:",pd.DataFrame(baggingReg_result,columns=['predicted_price']))

            gradientBoostingReg_result=gbr.predict(df)
            st.write("Результат GradientBoostingRegressor:",pd.DataFrame(gradientBoostingReg_result,columns=['predicted_price']))
        
            stackingReg_result=sr.predict(df)
            st.write("Результат StackingRegressor:",pd.DataFrame(stackingReg_result,columns=['predicted_price']))
        
            neoroReg_result=nn.predict(df)
            st.write("Результат нейронной сети:",pd.DataFrame(neoroReg_result,columns=['predicted_price']))


def visualization():

    st.title('Визуализация датасета')

    st.header("Тепловая карта корреляций числовых признаков")
    fig = plt.figure()
    fig.add_subplot(sns.heatmap(df[['price', 'area', 'latitude', 'longitude', 'bedrooms', 'bathrooms', 'balconyv']].corr(), annot=True))
    st.pyplot(fig)

    st.header("График BoxPlot цены дома")
    fig = plt.figure()
    plt.boxplot(df["price"])
    st.pyplot(plt)

    st.header("Гистограмма распределения площади домов")
    fig = plt.figure()
    fig.add_subplot(sns.histplot(df["area"]))
    st.pyplot(fig)

    st.header("Круговая диаграмма для типа здания")
    fig = plt.figure()
    size = df.groupby("type_of_building").size()
    plt.pie(size.values, labels=size.index, counterclock=True, autopct='%1.0f%%')
    st.pyplot(fig)

main()
