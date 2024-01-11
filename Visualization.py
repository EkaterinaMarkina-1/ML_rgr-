import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data= pd.read_csv("mumbai_houses_task.csv")
st.title('Визуализация датасета')

st.header('Датасет для классификации - "Опасен ли астероид"')

st.markdown('---')

st.write("Диаграмма с областями для скорости и расстояния")

chart_data = pd.DataFrame(np.random.randn(20, 2), columns=["relative_velocity", "miss_distance"])
st.area_chart(chart_data)

st.write("Диаграмма с областями для параметров скорости min и max")

chart_data = pd.DataFrame(np.random.randn(20, 2), columns=["est_diameter_max", "est_diameter_min"])
st.area_chart(chart_data)

st.write("Диаграмма рассеяния для скорости, расстояния и светимости")

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["relative_velocity", "miss_distance", "absolute_magnitude"])
st.scatter_chart(chart_data)
st.write("Гистограмма предсказываемого признака")

fig, ax = plt.subplots()
ax.hist(data['hazardous'], bins=20)

st.pyplot(fig)