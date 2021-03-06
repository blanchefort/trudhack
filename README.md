# Система определения предлагаемой заработной платы

Модель для предикта зарплаты по нескольким входящим параметрам:
* Название вакансии
* Тип занятости (код)
* Тип графика (код)
* Опыт (код)
* Город

Модель архитектуры [TabNet](https://arxiv.org/abs/1908.07442) обучалась на выборочных данных портала HeadHunter за 2019-2020 гг.

Сейчас модель делает предикт для некоторых городов России, но имеется возможность сделать так, чтобы модель определяла зарплату для вакансии любой точки России.

## Код
* [Ноутбук](https://github.com/blanchefort/trudhack/blob/master/notebooks/SALARY_01_TABNETL_PREPARE_DATA.ipynb), в котором отображён процесс предобработки данных
* [Ноутбук](https://github.com/blanchefort/trudhack/blob/master/notebooks/salary-tabnet-train.ipynb), в котором отображён процесс обучения модели
* [Файл app.py](https://github.com/blanchefort/trudhack/blob/master/app.py) содержит код, который загружает и инициализирует веса обученной модели и делает возможность обращаться к ней посредством REST API запросов.
* [Ноутбук](https://github.com/blanchefort/trudhack/blob/master/notebooks/usage_example.ipynb), в котором показано, как получать предикты с помощью API.

## Презентации
* [Презентация данного проекта](https://github.com/blanchefort/trudhack/blob/master/presentations/presentation_2.pdf)
* [Презентация идеи реализации сервиса для Кейса 1](https://github.com/blanchefort/trudhack/blob/master/presentations/presentation_1.pdf) - извлечение структурированной информации.

В файле [`notebooks/results.txt`](https://github.com/blanchefort/trudhack/blob/master/notebooks/results.txt) содержится пример результата работы модели. Там даны наименование вакансии, указанная зарплата и предсказанная моделью зарплата. Как видно, модель в ряде случаев несколько завышает зарплату по сравнению с официально заявленной для данной позиции. (Видимо, модель понимает реальное экономическое положение соискателей, и старается им помочь😻).

## Как запустить файл app.py у себя

Установка и запуск:

```
git clone https://github.com/blanchefort/trudhack.git
cd trudhack
python -m venv venv
pip install -r requirements.txt
```

Запуск:
```
python app.py
```
Или так:
```
uvicorn app:app --port 5000 --host 0.0.0.0
```

Документация API:

```
http://35.225.239.24:5000/docs
http://35.225.239.24:5000/redoc
```

## Контакты

```
Игорь Шаталин
+7 987 655 67 79
shatalin.ip@gmail.com
```

```
Андрей Лукин
andrew.luckin2015@yandex.ru
```
