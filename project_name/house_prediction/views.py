from django.shortcuts import render
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials, firestore, storage
import uuid
from .utils import generate_correlation_matrix_image
from .utils import generate_distribution_plot_image
from .utils import generate_pairplot_image, generate_trend_line_plot_image, predict, generate_pie_chart_image, generate_scatter_plot_image

from django.http import JsonResponse
from django.conf import settings
from django.http import FileResponse
import os
import json

# Load label encoders
with open(r'C:\Users\muniza.hashim\Desktop\senior\FYP\FYP progress\House Prices\Preprocessing Data\PredictHouse(Backend)\PredictHouse(Backend)\project_name\house_prediction\static\district_encoder.pkl', 'rb') as file5:
    district_encoder = pickle.load(file5)
with open(r'C:\Users\muniza.hashim\Desktop\senior\FYP\FYP progress\House Prices\Preprocessing Data\PredictHouse(Backend)\PredictHouse(Backend)\project_name\house_prediction\static\micro_district_encoder.pkl', 'rb') as file6:
    micro_district_encoder = pickle.load(file6)
with open(r'C:\Users\muniza.hashim\Desktop\senior\FYP\FYP progress\House Prices\Preprocessing Data\PredictHouse(Backend)\PredictHouse(Backend)\project_name\house_prediction\static\building_type_encoder.pkl', 'rb') as file7:
    building_type_encoder = pickle.load(file7)
with open(r'C:\Users\muniza.hashim\Desktop\senior\FYP\FYP progress\House Prices\Preprocessing Data\PredictHouse(Backend)\PredictHouse(Backend)\project_name\house_prediction\static\condition_encoder.pkl', 'rb') as file8:
    condition_encoder = pickle.load(file8)

house_data = pd.read_csv(r'C:\Users\muniza.hashim\Desktop\senior\FYP\FYP progress\House Prices\Preprocessing Data\PredictHouse(Backend)\PredictHouse(Backend)\project_name\house_prediction\static\house_kg_10K_ads.csv')

independent_variables = ["square", "rooms", "floors", "floor", "date_year"]
dependent_variable = "price"

# Define the categorical variables
categorical_variables = ['district', 'micro_district', 'building_type', 'condition']

# Create and fit the label encoders for each categorical variable
label_encoders = {var: LabelEncoder().fit(house_data[var]) for var in categorical_variables}

# Transform and add the encoded columns to house_data
for var, encoder in label_encoders.items():
    encoded_column = var + '_encoded'
    house_data[encoded_column] = encoder.transform(house_data[var])
    independent_variables.append(encoded_column)

# Sidebar - Dropdowns for categorical inputs and encoding
district_options = ['Октябрьский район', 'Ленинский район', 'Первомайский район', 'Свердловский район']
microdistrict_options = ['Магистраль', 'Академия Наук', 'ЖД вокзал', 'Unknown', 'Аламедин-1 м-н',
                         '6 м-н', 'Кок-Жар ж/м', 'Асанбай м-н', 'Джал-23 м-н (Нижний Джал)',
                         'Военторг', 'АЮ Grand', 'Восток-5 м-н', 'Молодая Гвардия',
                         'Верхний Джал м-н', 'КНУ', '4 м-н', 'Политех', 'Джал 15 м-н',
                         'Ипподром', 'Площадь Победы', '11 м-н', 'Мед. академия', 'Ак Кеме',
                         'Моссовет', 'Горького - Панфилова', 'Московская - Белинка',
                         'Старый аэропорт', 'АУЦА', 'Дворец спорта', '12 м-н', 'Гоин',
                         'Московская - Уметалиева', 'Парк Ататюрк', 'Жилгородок Ницца',
                         'Карла Маркса', 'ЦУМ', 'Сквер Тоголок Молдо', 'Бишкек-Парк',
                         'Душанбинка', 'Восточный автовокзал', 'Центральная мечеть',
                         'Юбилейка', 'Космос', '8 м-н', 'Кара-Жыгач ж/м',
                         'Джальская больница', 'Средний Джал м-н', 'Золотой квадрат',
                         'Ден Сяопина - Фучика', 'Нижний Токольдош', '5 м-н', 'Матросова',
                         'Парк Панфилова/Спартак', '7 м-н', 'Карпинка', 'Кудайберген',
                         'Джал-29 м-н', 'Улан м-н', 'Пишпек ж/м', 'ТЭЦ', 'БГУ', 'VEFA',
                         'Щербакова ж/м', 'Ак Эмир рынок', 'Госрегистр', 'Кок-Жар м-н',
                         'Церковь', 'Чуй - Алматинка', '3 м-н', 'Азия Молл',
                         'Цирк/Дворец бракосочетания', 'Шлагбаум', 'Филармония',
                         'Джал-30 м-н', 'Нижний Джал м-н', 'Джал 30', 'КГУСТА', '10 м-н',
                         'Западный автовокзал', 'Таатан', 'Гагарина', 'Учкун м-н',
                         'Тунгуч м-н', 'Городок энергетиков', '9 м-н',
                         'Советская - Скрябина', 'Ак-Орго ж/м', 'Арча-Бешик ж/м', 'Баха',
                         'Городок строителей', 'Вечерка', 'Кара Дарыя', 'Юг-2 м-н',
                         '1000 мелочей', 'Ген прокуратура', 'Таш Рабат',
                         'Аламединский рынок', 'Жилгородок Совмина ж/м', 'Улан-2 м-н',
                         'Дордой Плаза', '110 квартал ж/м', 'Кызыл-Аскер ж/м', 'Достук',
                         'Мадина', 'Ак-Босого ж/м', 'Алматинка - Магистраль', 'Комфорт',
                         'Ошский рынок', 'Токольдош ж/м', 'Киргизия 1 ж/м',
                         'Алтын-Ордо ж/м', 'Рухий Мурас ж/м', 'Достук м-н', 'Учкун ж/м',
                         'Орозбекова - Жибек-Жолу', 'Рабочий Городок', 'Ала-Арча ж/м',
                         'с. Орто-Сай', 'с. Чон-Арык', 'Дордой ж/м', 'Ак-Ордо ж/м',
                         'Колмо ж/м', 'Физкультурный', 'Эне-Сай ж/м', 'Киргизия-2 м-н',
                         'Ынтымак ж/м', 'Салам-Алик ж/м', 'Старый толчок',
                         'Балбан-Таймаш ж/м', 'Детский мир', '69-га', 'Учкун-2 ж/м',
                         'Касым ж/м', 'Ортосайский рынок',
                         'Красный Строитель 2 ж/м']  # Assume this contains the full list of microdistricts
building_type_options = ['кирпичный', 'монолитный', 'панельный']
condition_options = ['под самоотделку (ПСО)', 'хорошее', 'евроремонт', 'среднее', 'не достроено', 'требует ремонта',
                     'черновая отделка', 'свободная планировка']  # Assume this contains the full list of conditions


def detailed_analysis(request):
    if request.method == 'GET':
        # Extract the analysis type from the query parameters
        analysis_type = request.GET.get('analysis')

        # Your existing analysis logic goes here...
        if analysis_type == 'Pairplot':
            # Code for generating pairplot
            image_path = generate_pairplot_image(house_data, independent_variables, dependent_variable)
            if os.path.exists(image_path):
                relative_path = os.path.relpath(image_path, settings.MEDIA_ROOT)
                image_url = os.path.join(settings.MEDIA_URL, relative_path)
                return JsonResponse({'image_url': image_url})
            else:
                return JsonResponse({'error': 'Correlation matrix image not found'}, status=404)
        elif analysis_type == 'Correlation Matrix':
            image_path = generate_correlation_matrix_image(house_data, independent_variables, dependent_variable)
            if os.path.exists(image_path):
                relative_path = os.path.relpath(image_path, settings.MEDIA_ROOT)
                image_url = os.path.join(settings.MEDIA_URL, relative_path)
                return JsonResponse({'image_url': image_url})
            else:
                return JsonResponse({'error': 'Correlation matrix image not found'}, status=404)
        elif analysis_type == 'Price Distribution':
            # Code for generating distribution plot
            image_path = generate_distribution_plot_image(house_data, dependent_variable)
            if os.path.exists(image_path):
                relative_path = os.path.relpath(image_path, settings.MEDIA_ROOT)
                image_url = os.path.join(settings.MEDIA_URL, relative_path)
                return JsonResponse({'image_url': image_url})
            else:
                return JsonResponse({'error': 'Correlation matrix image not found'}, status=404)
        elif analysis_type == 'Price Trends':
            # Code for generating trend line plot
            image_path = generate_trend_line_plot_image(house_data, dependent_variable)
            if os.path.exists(image_path):
                relative_path = os.path.relpath(image_path, settings.MEDIA_ROOT)
                image_url = os.path.join(settings.MEDIA_URL, relative_path)
                return JsonResponse({'image_url': image_url})
            else:
                return JsonResponse({'error': 'Correlation matrix image not found'}, status=404)
        elif analysis_type == 'Price by District':
            # Code for generating pie chart
            image_path = generate_pie_chart_image(house_data, dependent_variable)
            if os.path.exists(image_path):
                relative_path = os.path.relpath(image_path, settings.MEDIA_ROOT)
                image_url = os.path.join(settings.MEDIA_URL, relative_path)
                return JsonResponse({'image_url': image_url})
            else:
                return JsonResponse({'error': 'Correlation matrix image not found'}, status=404)

    else:
        return JsonResponse({'error': 'Only GET requests are allowed'}, status=405)

def analyze_price(request):
    if request.method == 'GET':
        # Extract the analysis type from the query parameters
        analysis_by = request.GET.get('analysisBy')
        print(analysis_by)
        # Code for generating pairplot
        image_path = generate_scatter_plot_image(house_data, analysis_by, dependent_variable)
        if os.path.exists(image_path):
            relative_path = os.path.relpath(image_path, settings.MEDIA_ROOT)
            image_url = os.path.join(settings.MEDIA_URL, relative_path)
            return JsonResponse({'image_url': image_url})
        else:
            return JsonResponse({'error': 'Correlation matrix image not found'}, status=404)


def predict_price(request):
    if request.method == 'GET':
        formData = {}
        for key, value in request.GET.items():
            if key.startswith('formData[') and key.endswith(']'):
                inner_key = key.split('[')[-1][:-1]
                formData[inner_key] = value

        result = predict(formData)
        if "error" in result:
            return JsonResponse({"error": result["error"]}, status=400)
        else:
            return JsonResponse(result)