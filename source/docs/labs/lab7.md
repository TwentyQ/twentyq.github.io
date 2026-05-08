# Лабораторная работа №7: Классификация твитов по тональности с использованием методов обработки естественного языка

## Цель работы

Освоить основные методы предобработки текстовых данных для задачи классификации: токенизацию, удаление стоп-слов, лемматизацию, векторизацию (CountVectorizer, TF-IDF), а также обучить и сравнить несколько моделей машинного обучения (логистическая регрессия, XGBoost, случайный лес) для определения эмоциональной окраски твита.

---

## Задание

### Основная часть
1. Загрузить выборку твитов с метками тональности (положительные / отрицательные).

2. Провести первичный анализ данных и разделить выборку на обучающую и тестовую.

3. Реализовать базовую модель с использованием CountVectorizer (униграммы) и логистической регрессии.

4. Оценить качество модели с помощью метрик precision, recall, f1-score, accuracy.

5. Изучить влияние различных n-грамм (уни-, би-, три-, пентаграммы) на качество классификации.

6. Применить TF‑IDF векторизацию для разных n-грамм и сравнить результаты.

7. Реализовать предобработку с удалением стоп-слов и пунктуации, токенизацией с помощью word_tokenize.

8. Провести эксперимент с оставлением пунктуации и объяснить полученные результаты.

### Самостоятельная часть
1. Обучить альтернативные модели: XGBClassifier и RandomForestClassifier, сравнить их с логистической регрессией.

2. Для TF‑IDF векторизации вычислить classification_report для биграмм и триграмм, сравнить с униграммами и пентаграммами.

3. Сформулировать выводы о влиянии выбора векторизатора, n-грамм и модели на итоговое качество.

---

## Код и команды

### 1. Установка и импорт библиотек
```bash
# Установка pymorphy3 (для лемматизации)
!pip install pymorphy3

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from string import punctuation

nltk.download('punkt_tab')
nltk.download('stopwords')
```

### 2. Загрузка и подготовка данных
```bash
# Скачивание файлов с GitHub
!wget https://raw.githubusercontent.com/Gavroshe/RuTweetCorp/master/positive.csv
!wget https://raw.githubusercontent.com/Gavroshe/RuTweetCorp/master/negative.csv

# Загрузка в DataFrame
positive = pd.read_csv('positive.csv', sep=';', usecols=[3], names=['text'])
positive['label'] = 'positive'

negative = pd.read_csv('negative.csv', sep=';', usecols=[3], names=['text'])
negative['label'] = 'negative'

# Объединение
df = pd.concat([positive, negative])

# Разделение на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(df.text, df.label, random_state=42)
```

### 3. Базовая модель: CountVectorizer (униграммы) + LogisticRegression
```bash
vectorizer = CountVectorizer(ngram_range=(1,1))
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(x_train_vec, y_train)
pred = clf.predict(x_test_vec)

print(classification_report(y_test, pred))
```
Результаты:

| Класс    | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| negative | 0.76      | 0.77   | 0.76     | 27799   |
| positive | 0.78      | 0.76   | 0.77     | 28910   |
| accuracy |           |        | 0.77     | 56709   |

### 4. Эксперименты с n-граммами (CountVectorizer)
```bush
vectorizer_3 = CountVectorizer(ngram_range=(3,3))
x_train_3 = vectorizer_3.fit_transform(x_train)
x_test_3 = vectorizer_3.transform(x_test)
clf.fit(x_train_3, y_train)
print(classification_report(y_test, clf.predict(x_test_3)))
```
Точность упала до 0.65, F1 по negative = 0.57 - из-за разреженности признаков.

### 5. Предобработка: токенизация, стоп-слова, пунктуация
```bash
noise = stopwords.words('russian') + list(punctuation)

smart_vectorizer = CountVectorizer(ngram_range=(1,1),
                                   tokenizer=word_tokenize,
                                   stop_words=noise)
x_train_smart = smart_vectorizer.fit_transform(x_train)
x_test_smart = smart_vectorizer.transform(x_test)
clf.fit(x_train_smart, y_train)
print(classification_report(y_test, clf.predict(x_test_smart)))
```
Результат: accuracy = 0.78, macro F1 = 0.78 - улучшение за счёт удаления шума.

### 6. Эксперимент: оставление пунктуации (анализ артефакта)
```bash
vectorizer_punct = CountVectorizer(ngram_range=(1,1), tokenizer=word_tokenize)
# ... обучение ...
print(classification_report(y_test, pred))
```
Результат: accuracy = 1.00 (идеальная классификация).
Причина: в данных присутствует сильный сигнал - смайлик :D, который почти всегда соответствует положительному классу. Модель выучила этот простой паттерн.

### 7. Самостоятельная работа: альтернативные модели
#### XGBClassifier
```bash
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

xgb = XGBClassifier(learning_rate=0.1, n_estimators=1000,
                    max_depth=5, min_child_weight=3,
                    gamma=0.2, subsample=0.6,
                    colsample_bytree=1.0, objective='binary:logistic',
                    seed=27)
xgb.fit(x_train_smart, y_train_enc)
pred_xgb = xgb.predict(x_test_smart)
print(classification_report(y_test_enc, pred_xgb, target_names=['negative','positive']))
```
Результаты:

| Класс    | Precision | Recall | F1-score |
|----------|-----------|--------|----------|
| negative | 0.71      | 0.78   | 0.74     |
| positive | 0.76      | 0.69   | 0.73     |
| accuracy |           |        | 0.73     |
#### RandomForestClassifier
```bash
rf = RandomForestClassifier(n_estimators=300, max_depth=15,
                            min_samples_split=10, min_samples_leaf=5)
rf.fit(x_train_smart, y_train)
pred_rf = rf.predict(x_test_smart)
print(classification_report(y_test, pred_rf))
```
Результаты:

| Класс    | Precision | Recall | F1-score |
|----------|-----------|--------|----------|
| negative | 0.84      | 0.40   | 0.54     |
| positive | 0.61      | 0.93   | 0.74     |
| accuracy |           |        | 0.67     |

**Вывод по моделям:**

* Логистическая регрессия показывает лучший сбалансированный результат (F1 = 0.78).

* XGBoost немного уступает (0.73).

* Случайный лес плохо распознаёт негативные твиты (recall = 0.40) - переобучается на редких признаках.

### 8. TF‑IDF векторизация для биграмм и триграмм
#### Биграммы (2,2)
```bash
tfidf_bigram = TfidfVectorizer(ngram_range=(2,2))
x_train_tfidf = tfidf_bigram.fit_transform(x_train)
x_test_tfidf = tfidf_bigram.transform(x_test)
clf.fit(x_train_tfidf, y_train)
print(classification_report(y_test, clf.predict(x_test_tfidf)))
```
Результаты:

| Класс    | Precision | Recall | F1-score |
|----------|-----------|--------|----------|
| negative | 0.72      | 0.67   | 0.70     |
| positive | 0.70      | 0.75   | 0.73     |
| accuracy |           |        | 0.71     |
#### Триграммы (3,3)
```bash
tfidf_bigram = TfidfVectorizer(ngram_range=(2,2))
x_train_tfidf = tfidf_bigram.fit_transform(x_train)
x_test_tfidf = tfidf_bigram.transform(x_test)
clf.fit(x_train_tfidf, y_train)
print(classification_report(y_test, clf.predict(x_test_tfidf)))
```
Результаты:

| Класс    | Precision | Recall | F1-score |
|----------|-----------|--------|----------|
| negative | 0.72      | 0.46   | 0.56     |
| positive | 0.62      | 0.83   | 0.71     |
| accuracy |           |        | 0.65     |

А вот таблица сравнения n-грамм для TF‑IDF:

| n-граммы    | Accuracy | Macro F1 | Изменение F1 |
|-------------|----------|----------|--------------|
| униграммы   | 0.76     | 0.76     | –            |
| биграммы    | 0.71     | 0.71     | -0.05        |
| триграммы   | 0.65     | 0.64     | -0.12        |
| пентаграммы | 0.56     | 0.45     | -0.31        |

**Вывод:** униграммы дают наилучшее качество; увеличение n ведёт к разреженности и потере обобщающей способности.

### 8. Результаты

#### Сводная таблица качества моделей (на предобработанных данных, униграммы)

| Модель                 | Accuracy | Macro F1 |
|------------------------|----------|----------|
| LogisticRegression     | 0.78     | 0.78     |
| XGBClassifier          | 0.73     | 0.73     |
| RandomForestClassifier | 0.67     | 0.64     |

#### Лучшие параметры предобработки

- **Токенизация:** `word_tokenize`
- **Удаление стоп-слов русского языка и пунктуации**
- **Векторизатор:** `CountVectorizer` или `TfidfVectorizer` с `ngram_range=(1,1)`
- **Классификатор:** логистическая регрессия

## Ссылки на результат: [**Ноутбук GoogleColab**](https://colab.research.google.com/drive/1ba0_GJhTpV-3Z5aYvI6MQVEyR-OLe5JP?usp=sharing)

## Выводы
**В результате выполнения лабораторной работы:**

* Освоены основные этапы предобработки текстовых данных для задачи классификации тональности.

* Изучены методы векторизации (мешок слов, TF‑IDF) и влияние n-грамм на качество модели.

* Проведено сравнение трёх алгоритмов: логистическая регрессия, XGBoost, случайный лес.

* Установлено, что для данного набора твитов лучшей моделью является логистическая регрессия с униграммами и удалением стоп-слов (accuracy 0.78).

* Показано, что увеличение размера n-грамм (биграммы, триграммы) ухудшает качество из-за разреженности признаков.

* Обнаружен артефакт: смайлик :D является почти идеальным предиктором положительного класса, что приводит к переоценке качества при неаккуратной предобработке.

**Навыки, приобретенные в ходе работы:**

* Загрузка и объединение текстовых данных из CSV.

* Применение CountVectorizer и TfidfVectorizer с настройкой n-грамм.

* Токенизация текста с помощью NLTK.

* Удаление стоп-слов и пунктуации.

* Обучение и оценка нескольких моделей классификации.

* Анализ и интерпретация метрик precision, recall, f1-score.

**Дата выполнения:** 22.04.2026