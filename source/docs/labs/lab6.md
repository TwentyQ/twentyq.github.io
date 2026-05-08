# Лабораторная работа №6: Очистка и трансформация данных. pandas

## Цель работы

Освоение методов очистки и трансформации данных с использованием библиотеки pandas на примере реальных данных из Kaggle (набор данных о пассажирах «Титаника»).

---

## Задание

1. Загрузить данные из CSV-файла, вывести первые 10 строк.

2. Проверить типы данных и количество пропусков в каждом столбце.

3. Получить статистические характеристики числовых признаков.

4. Построить гистограммы распределения для числовых признаков.

5. Обработать пропуски:

> Age - заполнить медианой, создать признак Age_group.

> Embarked - заполнить модой.

> Cabin - создать признак Cabin_letter (первая буква или Unknown).

6. Преобразовать типы данных:

> Pclass - категориальный Pclass_cat (1-F, 2-S, 3-T).

> Из Name выделить Title (Mr, Mrs, Miss, Rare...).

> Sex - числовой Sex_binary (0/1).

> Создать FamilySize = SibSp + Parch + 1 и IsAlone.

7. Удалить выбросы:

> Построить boxplot для Fare, применить winsorization (95-й перцентиль).

> Аналогично для Age.

8. Выполнить агрегацию:

> Среднее выживание по классам, по полу и классу.

> Медианный возраст по портам посадки.

> Сводную таблицу выживаемости по новым признакам.

9. Сохранить очищенные данные в CSV.

10. Вычислить метрики качества очистки: процент заполненных пропусков, количество уникальных значений, корреляцию новых признаков.

---

## Код и команды

### 1. Установка и импорт библиотек
```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### 2. Загрузка данных
```bash
!wget https://raw.githubusercontent.com/TwentyQ/for_different_things/refs/heads/main/train.csv
!wget https://raw.githubusercontent.com/TwentyQ/for_different_things/refs/heads/main/test.csv

training_data = pd.read_csv('train.csv')
```

### 3. Первичный анализ
```bash
# Первые 10 строк
training_data.head(10)

# Типы данных
training_data.info()

# Количество пропусков
training_data.isnull().sum()

# Статистика числовых признаков
training_data.describe().T

# Гистограммы
training_data[['Age','Fare','SibSp','Parch']].hist(bins=30, figsize=(12,8))
plt.show()
```

### 4. Обработка пропусков

**Age** - заполнение медианой

```bush
median_age = training_data['Age'].median()   # 28.0
training_data['Age'] = training_data['Age'].fillna(median_age)

# Создание возрастной группы
bins = [0,12,18,35,60,100]
labels = ['Child','Teen','Young Adult','Adult','Senior']
training_data['Age_group'] = pd.cut(training_data['Age'], bins=bins, labels=labels, right=False)
```

**Embarked ** - заполнение модой

```bush
mode_embarked = training_data['Embarked'].mode()[0]   # 'S'
training_data['Embarked'].fillna(mode_embarked, inplace=True)
```

**Cabin ** - извлечение первой буквы

```bush
training_data['Cabin_letter'] = np.where(
    training_data['Cabin'].isna(),
    'Unknown',
    training_data['Cabin'].str[0]
)
```

### 5. Трансформация данных
```bash
# Pclass в категориальный
pclass_map = {1:'F', 2:'S', 3:'T'}
training_data['Pclass_cat'] = training_data['Pclass'].map(pclass_map)

# Title из имени
training_data['Title'] = training_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
title_map = {
    'Mr':'Mr', 'Miss':'Miss', 'Mrs':'Mrs', 'Master':'Master',
    'Dr':'Rare', 'Rev':'Rare', 'Col':'Rare', 'Major':'Rare',
    'Mlle':'Miss', 'Ms':'Miss', 'Lady':'Rare', 'Countess':'Rare',
    'Jonkheer':'Rare', 'Don':'Rare', 'Dona':'Rare', 'Mme':'Mrs',
    'Capt':'Rare', 'Sir':'Rare'
}
training_data['Title'] = training_data['Title'].map(title_map).fillna('Rare')

# Sex в числовой
training_data['Sex_binary'] = (training_data['Sex'] == 'female').astype(int)

# Семейные признаки
training_data['FamilySize'] = training_data['SibSp'] + training_data['Parch'] + 1
training_data['IsAlone'] = (training_data['FamilySize'] == 1).astype(int)
```

### 6. Обработка выбросов (winsorization)
```bash
# Fare – 95-й перцентиль
p95_fare = training_data['Fare'].quantile(0.95)
training_data['Fare_winsorized'] = training_data['Fare'].clip(upper=p95_fare)

# Age – 95-й перцентиль
p95_age = training_data['Age'].quantile(0.95)
training_data['Age_winsorized'] = training_data['Age'].clip(upper=p95_age)
```

### 7. Агрегация и анализ
```bash
# Средняя выживаемость по классу
training_data.groupby('Pclass')['Survived'].mean()

# По классу и полу
training_data.groupby(['Pclass','Sex'])['Survived'].mean()

# Медианный возраст по портам
training_data.groupby('Embarked')['Age'].median()

# Сводная таблица
pd.pivot_table(training_data, values='Survived', index='Pclass_cat', columns='Age_group', aggfunc='mean')
```

### 8. Сохранение результата
```bash
training_data.to_csv('titanic_cleaned.csv', index=False)
```

### 9. Результаты
### Таблица пропусков до обработки

| Столбец   | Пропуски | % от всех записей |
|-----------|----------|-------------------|
| Age       | 177      | 19.9%             |
| Embarked  | 2        | 0.2%              |
| Cabin     | 687      | 77.1%             |

После обработки: Age и Embarked заполнены на 100%, для Cabin создан новый признак.

### Выбросы (по IQR)

| Признак | Количество выбросов | %    | Границы (нижняя – верхняя) |
|---------|---------------------|------|----------------------------|
| Fare    | 116                 | 13.0%| -26.72 – 65.63             |
| Age     | 66                  | 7.4% | 2.50 – 54.50               |

После winsorization (95-й перцентиль):

* Fare ограничено сверху значением 112.08

* Age ограничено сверху значением 57.0

### Средняя выживаемость по классу билета

| Pclass | Выживаемость |
|--------|--------------|
| 1      | 0.630        |
| 2      | 0.473        |
| 3      | 0.242        |

### Средняя выживаемость по классу и полу

| Pclass | Sex    | Выживаемость |
|--------|--------|--------------|
| 1      | female | 0.968        |
| 1      | male   | 0.369        |
| 2      | female | 0.921        |
| 2      | male   | 0.157        |
| 3      | female | 0.500        |
| 3      | male   | 0.135        |

### Корреляция новых признаков

|                 | FamilySize | IsAlone | Sex_binary | Age_winsorized | Fare_winsorized |
|-----------------|------------|---------|------------|----------------|-----------------|
| FamilySize      | 1.00       | -0.81   | -0.02      | 0.11           | 0.12            |
| IsAlone         | -0.81      | 1.00    | -0.02      | -0.06          | -0.06           |
| Sex_binary      | -0.02      | -0.02   | 1.00       | -0.05          | -0.19           |
| Age_winsorized  | 0.11       | -0.06   | -0.05      | 1.00           | 0.19            |
| Fare_winsorized | 0.12       | -0.06   | -0.19      | 0.19           | 1.00            |

## Ссылки на результат: [**Ноутбук GoogleColab**](https://colab.research.google.com/drive/1yijSHBR-zLEs0-DRl1YbcE9PRFLjG-GD?usp=sharing) 

## Выводы
**В результате выполнения лабораторной работы:**

* Освоены основные методы очистки данных: выявление и заполнение пропусков (медиана, мода), создание новых признаков на основе существующих.

* Проведена трансформация типов данных и категориальных переменных.

* Применена winsorization для борьбы с выбросами – сохранены все наблюдения, но уменьшено влияние экстремальных значений.

* Выполнена агрегация данных и построены сводные таблицы, выявляющие закономерности выживаемости (женщины и пассажиры первого класса выживали чаще).

* Рассчитаны метрики качества очистки: 100% заполнение критических пропусков, информативные категориальные признаки, разумная корреляция между новыми переменными.

**Навыки, приобретенные в ходе работы:**

* Загрузка и первичный анализ данных в pandas.

* Визуализация пропусков и распределений (matplotlib, seaborn).

* Заполнение пропусков (median, mode) и создание производных признаков.

* Обработка выбросов методом winsorization.

* Группировка, агрегация и создание сводных таблиц.

* Оценка качества очистки данных.

**Дата выполнения:** 22.04.2026