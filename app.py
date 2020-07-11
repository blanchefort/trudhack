import os
import numpy as np
import math
import pickle
import typing
from typing import List, Tuple, Optional
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from transformers import BertTokenizer
from transformers import BertModel
from google_drive_downloader import GoogleDriveDownloader as gdd
import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel


device = torch.device('cpu')
base_path = os.path.abspath(os.path.dirname(__file__))
system_path = os.path.join(base_path, 'system_files')

def load_config_data(system_path: str) -> Tuple:
    """Загружаем словари, использовавшиеся при обучениии модели:
    - experience2id
    - schedule2id
    - employment2id
    - city2id - названия городов
    - salary_SF_mydata_600k - массив зарплат, для перевода нормализованного значения в реальное
    """
    if not os.path.isfile(os.path.join(system_path, 'salary_experience2id.pickle')):
        gdd.download_file_from_google_drive(file_id='158xkRQ2Y8o3RZjDtupvwG458LxrTjZ5x',
                                            dest_path=os.path.join(system_path, 'salary_experience2id.pickle'),
                                            unzip=False)
    if not os.path.isfile(os.path.join(system_path, 'salary_schedule2id.pickle')):
        gdd.download_file_from_google_drive(file_id='1-8If6uch9w_wvhe1FLc0AvKqBKBkFZDc',
                                            dest_path=os.path.join(system_path, 'salary_schedule2id.pickle'),
                                            unzip=False)
    if not os.path.isfile(os.path.join(system_path, 'salary_employment2id.pickle')):
        gdd.download_file_from_google_drive(file_id='1-Dqd2f1Pa-6fOrq1agPUewrP3XcikXJ6',
                                            dest_path=os.path.join(system_path, 'salary_employment2id.pickle'),
                                            unzip=False)
    if not os.path.isfile(os.path.join(system_path, 'salary_city2id.pickle')):
        gdd.download_file_from_google_drive(file_id='1-98ual4XOChTBsyp-QVNcoBO2C3hgovd',
                                            dest_path=os.path.join(system_path, 'salary_city2id.pickle'),
                                            unzip=False)
    if not os.path.isfile(os.path.join(system_path, 'salary_SF_mydata_600k.npy')):
        gdd.download_file_from_google_drive(file_id='1-8XhepvRFOWYZXVUhF-vBBBDFs4tGU5X',
                                            dest_path=os.path.join(system_path, 'salary_SF_mydata_600k.npy'),
                                            unzip=False)
    
    with open(os.path.join(system_path, 'salary_experience2id.pickle'), 'rb') as handle:
        experience2id = pickle.load(handle)

    with open(os.path.join(system_path, 'salary_schedule2id.pickle'), 'rb') as handle:
        schedule2id = pickle.load(handle)

    with open(os.path.join(system_path, 'salary_employment2id.pickle'), 'rb') as handle:
        employment2id = pickle.load(handle)

    with open(os.path.join(system_path, 'salary_city2id.pickle'), 'rb') as handle:
        city2id = pickle.load(handle)

    salary_orig = np.load(os.path.join(system_path, 'salary_SF_mydata_600k.npy'))

    return experience2id, schedule2id, employment2id, city2id, salary_orig

def load_tabnet(system_path: str) -> TabNetRegressor:
    """Загрузка и инициализация предобученной модели TabNet
    """
    if not os.path.isfile(os.path.join(system_path, 'SalaryTabnetL_GPU.pt')):
        gdd.download_file_from_google_drive(file_id='1KXdys6uHlkjhsT5-ZfjAGeUbE30UTtJj',
                                            dest_path=os.path.join(system_path, 'SalaryTabnetL_GPU.pt'),
                                            unzip=False)
    tabnet = torch.load(os.path.join(system_path, 'SalaryTabnetL_GPU.pt'), map_location=torch.device('cpu'))
    tabnet.device = 'cpu'
    tabnet.network.to(device);
    return tabnet

def get_salary(orig: Optional, scaled: Optional) -> Optional:
    """Переводит нормализованное значение зарплаты в реальное
    """
    return (scaled * orig.std(axis=0)) + orig.mean(axis=0)

def get_scaled_salary(orig: Optional, salary: Optional) -> Optional:
    """Возвращает нормализованное значение зарплаты
    """
    return (salary - orig.mean(axis=0))  / orig.std(axis=0)

def round_salary(salary: float) -> float:
    """Округляет значение зарплаты до десятков тысяч
    """
    return math.floor(salary / 1000) * 1000

# Метод округления, применяемый ко всему numpy-массиву
round_salary_array = np.vectorize(round_salary)

def prepare_names(names: List[str]) -> List:
    """Создаёт из текстового представления профессии её усреднённое векторное представление
    """
    global bert, tokenizer
    step=1000
    names_vectors = []
    for batch_i in range(0, len(names), step):
        inputs = tokenizer(
            names[batch_i:batch_i+step],
            padding='max_length',
            truncation=True,
            max_length=20,
            return_tensors='pt',
            return_attention_mask=True,
        )
        with torch.no_grad():
            output = bert(**inputs)
        total_sum = torch.zeros_like(output[0][:, 1, :])
        for i in range(0, output[0].shape[1]):
            total_sum += output[0][:, i, :]
        total_sum = total_sum / output[0].shape[1]
        names_vectors.extend(total_sum.detach().numpy())

        return names_vectors

def prepare_meta(
    experience: List[str],
    schedule: List[str],
    employment: List[str],
    city: List[str]) -> Tuple:
    """Преобразует мета-параметры из текстовых значений в индексы
    """
    global experience2id, schedule2id, employment2id, city2id

    experience = [experience2id[item] for item in experience]
    schedule = [schedule2id[item] for item in schedule]
    employment = [employment2id[item] for item in employment]
    city = [city2id[item] for item in city]

    experience = np.array(experience)
    schedule = np.array(schedule)
    employment = np.array(employment)
    city = np.array(city)

    experience = np.reshape(experience, (experience.shape[0], 1))
    schedule = np.reshape(schedule, (schedule.shape[0], 1))
    employment = np.reshape(employment, (employment.shape[0], 1))
    city = np.reshape(city, (city.shape[0], 1))

    return experience, schedule, employment, city

def evaluate_predict(
    names: List[str],
    experience: List[str],
    schedule: List[str],
    employment: List[str],
    city: List[str],
    rounded: bool=True) -> Optional:
    """Возвращает предполагаемые зарплаты
    """
    global tabnet, salary_orig
    global experience2id, schedule2id, employment2id, city2id

    if len(names) == len(experience) == len(schedule) == len(employment) == len(city) and len(names) > 0:
        for item in experience:
            if item not in experience2id.keys():
                return False
        for item in schedule:
            if item not in schedule2id.keys():
                return False
        for item in employment:
            if item not in employment2id.keys():
                return False
        for item in city:
            if item not in city2id.keys():
                return False

        names = prepare_names(names)
        experience, schedule, employment, city = prepare_meta(experience, schedule, employment, city)

        X = np.concatenate((names, experience, schedule, employment, city), axis=1)
        y = tabnet.predict(X)
        salary = get_salary(orig=salary_orig, scaled=y)
        if rounded == True:
            salary = round_salary_array(salary)
        return salary
    return False

# load config data
experience2id, schedule2id, employment2id, city2id, salary_orig = load_config_data(system_path)

# load models
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
bert = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')
tabnet = load_tabnet(system_path)

#run

class InputData(BaseModel):
    """Формат входящих данных
    """
    names: List[str] = []
    experience: List[str] = []
    schedule: List[str] = []
    employment: List[str] = []
    city: List[str] = []

app = FastAPI(title='Система определения предлагаемой заработной платы', version='0.1.0')

@app.get('/predict', description='Получаем предикты зарплат')
async def predict(data: InputData):
    """Получаем предикты зарплат
    """
    global tabnet, salary_orig, round_salary_array
    global experience2id, schedule2id, employment2id, city2id

    names = data.names
    experience = data.experience
    schedule = data.schedule
    employment = data.employment
    city = data.city
    names = data.names

    if len(names) == len(experience) == len(schedule) == len(employment) == len(city) and len(names) > 0:
        for item in experience:
            if item not in experience2id.keys():
                raise HTTPException(status_code=422, detail='experience not in list')
        for item in schedule:
            if item not in schedule2id.keys():
                raise HTTPException(status_code=422, detail='schedule not in list')
        for item in employment:
            if item not in employment2id.keys():
                raise HTTPException(status_code=422, detail='employment not in list')
        for item in city:
            if item not in city2id.keys():
                raise HTTPException(status_code=422, detail='city not in list')

        names = prepare_names(names)
        experience, schedule, employment, city = prepare_meta(experience, schedule, employment, city)

        X = np.concatenate((names, experience, schedule, employment, city), axis=1)
        y = tabnet.predict(X)
        salary = get_salary(orig=salary_orig, scaled=y)
        # if rounded == True:
        #     salary = round_salary_array(salary)
        return {'result': round_salary_array(salary)}
    else:
        raise HTTPException(status_code=422, detail='bad request')


if __name__ == '__main__':
    uvicorn.run(app)