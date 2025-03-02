import pandas as pd
import streamlit as st
from functions import *


input_fetures = dict()


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


# Заголовок сторінки
st.title(":blue[Застосунок класифікації відтоку клієнтів]")
st.subheader("Введіть значення ознак для класифікаціїї окремих клієнтів")

input_fetures['is_tv_subscriber'] = st.selectbox(
    'is_tv_subscriber', options=[0, 1])
input_fetures['is_movie_package_subscriber'] = st.selectbox(
    'is_movie_package_subscriber', options=[0, 1])
for name, val in features_range.items():
    if name.startswith('is'):
        continue
    input_fetures[name] = st.slider(
        name, min_value=min(val), max_value=max(val))


selectore = st.sidebar.selectbox(
    'Оберіть модель класифікатора', ["RandomForest Classificator",
                                     "Logistic regression Classificator",
                                     "SGD Classificator",
                                     "Ansamble ML Classificator",
                                     "Neural Network Classificator"])
st.sidebar.info(
    f"Точність класифікатора {class_accuracy[selectore]}%")

is_percent = st.sidebar.checkbox('Додати відсоток вирогідності прогнозу')

if st.button(label='Зробити передбачення ', key='predict_one'):

    one_dataframe = pd.DataFrame()
    for key, val in input_fetures.items():
        one_dataframe[key] = [val]
    one_predict = get_predictions(one_dataframe, model_choice=selectore,
                                  percents=is_percent)
    pred = one_predict.to_dict()
    if pred.get('churn')[0] == 1:
        st.subheader(f"Результат прогнозування: Клієнт піде")
    else:
        st.subheader(f"Результат прогнозування: Клієнт залишиться")
    if pred.get('churn_percent'):
        st.subheader(
            f"Ймовірність: {pred.get('churn_percent')[0]}%")


st.sidebar.header(
    "Завантажте csv-файл з данними клієнтів")
uploaded_file = st.sidebar.file_uploader(
    "Виберіть файл...", help='Прожмакай кнопку!', type=["csv", "CSV"])

if uploaded_file is not None:
    # Відкриття файла з данними
    try:
        predict_dataframe = False
        dataframe = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Завантажено файл: {uploaded_file.name}")

        if st.sidebar.button(
                label='Зробити передбачення', key='predict'):

            predict_dataframe = get_predictions(dataframe,
                                                model_choice=selectore,
                                                percents=is_percent)

            st.sidebar.success(f"Прогноз успішно зроблено.")

            csv_data = convert_df(predict_dataframe)
            file_name = f"{uploaded_file.name[:-4]}_{class_file_ext[selectore]}.csv"

            st.sidebar.download_button(
                label="Download data as CSV", data=csv_data, file_name=file_name, mime="text/csv")
    except Exception as e:
        st.sidebar.error(f"ERROR:\n\r{e}")
