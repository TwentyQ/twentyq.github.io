# Лабораторная работа №8: Веб-скрапинг

## Цель работы

Освоение методов скрапинга (парсинга) данных из веб-страниц с использованием Python, получение практических навыков работы с библиотеками requests, BeautifulSoup, а также сохранение структурированных данных в CSV-файлы.

---

## Задание

### Основная часть
1. Создать в Google Colab ноутбук для выполнения скрапинга.
2. Выбрать объект скрапинга – сайт https://news.itmo.ru/ru.
3. Спарсить для каждой новости (из поиска по ключевым словам): 
> идентификатор новости (целочисленное число из URL);

> название новости;

> дату её размещения;

> URL на страницу с конкретной новостью.

4. Сохранить полученные данные в CSV-файл (scraped_news_data.csv).

### Самостоятельная часть
1. Для каждой новости из полученного списка до парсить:

> количество просмотров;

> текст новости;

> теги.

2. Сохранить детальные данные в CSV-файл (news_content/news_details.csv) в отдельной папке news_content.

3. Обеспечить корректную обработку ошибок и паузы между запросами.

4. Вывести первые строки обоих CSV для проверки.

---

## Код и команды

### 1. Импорт библиотек
```bash
import csv
import requests
from bs4 import BeautifulSoup
import re
import time
import os
import random
import pandas as pd
```

### 2. Константы и списки для сбора данных
```bash
DOMAIN = 'https://news.itmo.ru'
SEARCH_DOMAIN = 'https://news.itmo.ru/ru/search/?search='

headers = []   # заголовки новостей
urls = []      # полные URL новостей

# Ключевые слова для поиска
queries = ['нейротехнологии', 'нейротехнологии и программирование']
```

### 3. Парсинг страниц поиска
```bash
for query in queries:
    data = requests.get(SEARCH_DOMAIN + query)
    bs = BeautifulSoup(data.text, 'html.parser')
    news_headers = bs.find_all('li', {'class': 'weeklyevent'})
    for _n in news_headers:
        post = _n.find('h4').find('a').get_text()
        headers.append(post)
        post_url = _n.find('h4').find('a')['href']
        urls.append(DOMAIN + post_url)

print(f"Всего найдено новостей: {len(urls)}") 
```

### 4. Сбор идентификатора и даты для каждой новости
```bush
news_data = []
output_csv_file = 'scraped_news_data.csv'

for i, (header, url) in enumerate(zip(headers, urls)):
    # Извлечение ID из URL
    match = re.search(r'/(\d+)/?$', url)
    news_id = match.group(1) if match else 'N/A'

    # Загрузка страницы новости
    time.sleep(random.randint(0, 1))
    response = requests.get(url)
    soup_news = BeautifulSoup(response.text, 'html.parser')
    time_tag = soup_news.find('time')
    news_date = time_tag.contents[0].strip() if time_tag else "Дата не найдена"

    news_data.append({
        'Идентификатор новости': news_id,
        'Название новости': header,
        'Дата её размещения': news_date,
        'URL на страницу с конкретной новостью': url
    })
```

### 5. Запись общего CSV
```bash
with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
        'Идентификатор новости',
        'Название новости',
        'Дата её размещения',
        'URL на страницу с конкретной новостью'
    ])
    writer.writeheader()
    writer.writerows(news_data)
print(f"Данные успешно записаны в {output_csv_file}")
```

### 6. Функция детального парсинга (просмотры, текст, теги)
```bash
def parse_news_details(url):
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        # Просмотры
        views_span = soup.find('span', class_='icon eye')
        views = views_span.get_text(strip=True) if views_span else "0"

        # Текст новости
        text_div = soup.find('div', class_='post-content')
        text = text_div.get_text(separator=' ', strip=True) if text_div else ""

        # Теги
        tags = []
        tags_ul = soup.find('ul', class_='tags')
        if tags_ul:
            for a in tags_ul.find_all('a'):
                tag = a.get_text(strip=True)
                if tag:
                    tags.append(tag)
        tags_str = ', '.join(tags)

        return views, text, tags_str
    except Exception as e:
        print(f"Ошибка детального парсинга {url}: {e}")
        return "0", "", ""
```

### 7. Сбор детальной информации и сохранение в папку news_content
```bash
os.makedirs('news_content', exist_ok=True)
details_csv = 'news_content/news_details.csv'

detailed_data = []
for i, item in enumerate(news_data):
    views, text, tags = parse_news_details(item['URL на страницу с конкретной новостью'])
    detailed_data.append({
        'id': item['Идентификатор новости'],
        'title': item['Название новости'],
        'date': item['Дата её размещения'],
        'views': views,
        'text': text,
        'tags': tags
    })
    time.sleep(0.7)   # вежливая пауза

with open(details_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'title', 'date', 'views', 'text', 'tags'])
    writer.writeheader()
    writer.writerows(detailed_data)

print(f"Детальный CSV сохранён как {details_csv}")
```

### 8. Проверка результатов с помощью pandas
```bash
print("Общий CSV")
df_common = pd.read_csv('scraped_news_data.csv')
display(df_common.head(3))

print("Детальный CSV")
df_detail = pd.read_csv('news_content/news_details.csv')
display(df_detail.head(3))
```
## Ссылки на результат: [**Ноутбук Google Colab**](https://colab.research.google.com/drive/1gsqpeAucFyXpV5CCNVMBj6twbUb53e7g?usp=sharing)

## Выводы
**В результате выполнения лабораторной работы:**

* Освоены базовые методы веб-скрапинга с использованием библиотек requests и BeautifulSoup.

* На практике изучено извлечение данных из HTML-структуры: поиск по тегам, классам, атрибутам.

* Реализована обработка нескольких страниц (поиск по ключевым словам), извлечение идентификатора через регулярные выражения.

* Организовано сохранение данных в двух форматах: общий список новостей и детальный набор (просмотры, текст, теги).

* Настроены паузы между запросами для снижения нагрузки на сервер и обработка ошибок.

**Навыки, приобретенные в ходе работы:**

* Работа с HTTP-запросами в Python.

* Парсинг HTML с BeautifulSoup.

* Использование регулярных выражений для извлечения чисел из URL.

* Запись словарей в CSV с помощью csv.DictWriter.

* Организация файловой структуры (создание папки, сохранение файлов).

* Применение библиотеки pandas для первичного просмотра данных.

**Дата выполнения:** 05.05.2026