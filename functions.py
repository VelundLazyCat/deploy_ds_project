import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import sklearn


model_tree = 'models/model_tree.pkl'
model_log = 'models/model_log.pkl'
model_sgd = 'models/model_sgd.pkl'
model_stack = 'models/stacking_model_ml.pkl'
model_nn = 'models/model_NN.keras'

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


def save_data(dataframe, name):
    dataframe.to_csv(name, index=False)


if __name__ == "__main__":

    # print('-'*20)
    # means, stds, eps = getmodel_ml(values)
    # model = getmodel_keras(model_nn)
    dataframe = pd.read_csv(
        r'C:\PythonProject\PythonCoreCourse\PythonDataSience\project\internet_service_churn.csv')

    # model = getmodel_ml(model_tree)

    pred_data = get_predictions(dataframe=dataframe, model_choice="Ansamble ML Classificator",
                                percents=True)

    print(pred_data.head())

    # pred_data.to_csv('internet_service_churn_pred.csv', index=False)
