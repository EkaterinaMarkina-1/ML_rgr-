import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("mumbai_houses_task.csv")
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
sns.set_palette("area")
plt.pie(size.values, labels=size.index, counterclock=True, autopct='%1.0f%%')
st.pyplot(fig)
