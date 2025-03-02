import pandas as pd
import streamlit as st
import numpy as np
#import tensorflow as tf
import joblib
import sklearn

model_tree = 'models/model_tree.pkl'
model_log = 'models/model_log.pkl'
model_sgd = 'models/model_sgd.pkl'
model_stack = 'models/stacking_model_ml.pkl'
model_nn = 'models/stacking_model_ml.pkl'

values = 'models/values.pkl'

accuracy_ml = joblib.load('models/accuracy_ml.pkl')
accuracy_nn = round(joblib.load('models/accuracy_nn.pkl'), 4)
accuracy_log = accuracy_ml['logreg']
accuracy_tree = accuracy_ml['ranfree']
accuracy_sgd = accuracy_ml['sgd']
accuracy_steck = accuracy_ml['stack']

features_range = {'is_tv_subscriber': [0, 1],
                  'is_movie_package_subscriber': [0, 1],
                  'subscription_age': [0.0, 12.8],
                  'bill_avg': [0.0, 406.0],
                  'reamining_contract': [0.0, 2.92],
                  'service_failure_count': [0.0, 19.0],
                  'download_avg': [0.0, 4415.2],
                  'upload_avg': [0.0, 453.3],
                  'download_over_limit': [0.0, 7.0]}

Classificator = {"RandomForest Classificator": model_tree,
                 "Logistic regression Classificator": model_log,
                 "SGD Classificator": model_sgd,
                 "Ansamble ML Classificator": model_stack,
                 "Neural Network Classificator": model_nn}

class_file_ext = {"RandomForest Classificator": 'rf',
                  "Logistic regression Classificator": 'lg',
                  "SGD Classificator": 'sgd',
                  "Ansamble ML Classificator": 'ans',
                  "Neural Network Classificator": 'nn'}

class_accuracy = {"RandomForest Classificator": accuracy_tree,
                  "Logistic regression Classificator": accuracy_log,
                  "SGD Classificator": accuracy_sgd,
                  "Ansamble ML Classificator": accuracy_steck,
                  "Neural Network Classificator": accuracy_nn}


def getmodel_ml(model):
    """Завантажує ML- модель з pkl файла
    потребує ім'я моделі"""

    return joblib.load(model)


def getmodel_keras(model):
    """Завантажує  модель нейронної мережі з файла .keras, фбо .h5
    потребує ім'я моделі"""

    return tf.keras.models.load_model(model)


def sanitazing_data(dataframe):
    """"Функція яка заповнює пусті Nan в колонці 'reamining_contract'
    і видаляє строки з нечисловими даними в інших стовпцях,
    побудована на базі EDA дослідження 
    повертає ощищений, зменшений датафрейм"""

    dataframe = dataframe.fillna(
        {'reamining_contract': dataframe['reamining_contract'].min()}).dropna()
    return dataframe


def get_stadartization_values(values):
    """Функція що завантажує константи для стандартизації данних з зовнішнього .pkl файлу
    Повертає кортеж ознак """

    return joblib.load(values)


def standartization(inputs):
    """ Загальна функція стандартизації вхідних ознак
    потребує глобальну змінну в якій зберігається ім'я файла з потрібними константами
    Повертає стандартизований масив ознак"""

    means, stds, eps = get_stadartization_values(values)
    return (inputs - means) / (stds + eps)


def preprocessing(dataframe):
    """Функція  підготовки даних. гарантує що порядок ознак буде той самий що був при тренуванні
    моделей. проводить стандартизацію вхідних данних.
    Приймає датафрейм pandas.
    Повертає стандартизований масив потрібних ознак"""

    important_features = ['is_tv_subscriber', 'is_movie_package_subscriber',
                          'subscription_age', 'bill_avg',
                          'reamining_contract', 'service_failure_count',
                          'download_avg', 'upload_avg',
                          'download_over_limit']

    dataframe = dataframe[important_features]
    dataframe = standartization(dataframe)

    return dataframe


def churn_predict_neural(inputs, model) -> tuple:
    """"Функція прогнозування классу користувача. приймае нормалізований масив даних,
    і модель нейронної мережі .
    Видае кортеж массивів з класифікацією і відсотком вирогідності классу"""

    predictions = model.predict(inputs, verbose=1)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_percent = np.round(np.max(predictions, axis=1), decimals=2)

    return predicted_class, predicted_percent


def churn_predict_ml(inputs, model) -> tuple:
    """"Функція прогнозування классу користувача. приймае нормалізований масив даних,
    і модель ML-класифікатора.
    Видае кортеж массивів з класифікацією і відсотком вирогідності классу"""

    predictions = model.predict_proba(inputs)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_percent = np.round(np.max(predictions, axis=1), decimals=2)

    return predicted_class, predicted_percent


def get_predictions(dataframe, model_choice, percents=False):
    """Головна функція по передбаченню.
    приймаэ датафрейм pandas, та вибір моделі користувача.
    повертає новий датафрейм pandas звибраними передбаченнями і 
    їх вирогідністю за флагом percents."""

    dataframe = sanitazing_data(dataframe)
    inputs_standard = preprocessing(dataframe)

    if 'Neural' in model_choice:
        model = getmodel_keras(Classificator[model_choice])
        predictions, predict_percents = churn_predict_neural(
            inputs=inputs_standard, model=model)
    else:
        model = getmodel_ml(Classificator[model_choice])
        predictions, predict_percents = churn_predict_ml(
            inputs=inputs_standard, model=model)

    dataframe['churn'] = predictions
    if percents:
        dataframe['churn_percent'] = predict_percents
    data = pd.DataFrame(dataframe)
    return data


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
