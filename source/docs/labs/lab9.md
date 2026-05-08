# Лабораторная работа №9: Работа с графикой. SourceCraft. CI/CD. Артефакты

## Цель работы

Освоить анализ и визуализацию данных с использованием библиотек pandas и matplotlib, научиться обрабатывать датасеты, создавать новые признаки, выполнять группировки и агрегации, строить информативные графики. Дополнительно — освоить основы CI/CD на платформе SourceCraft: настроить автоматическое выполнение Jupyter Notebook, генерацию HTML-отчёта и сохранение артефактов с меткой о выполнении в среде CI.

---

## Задание

### Основная часть
1. Сделать форк репозитория с заданием на SourceCraft, получить собственный репозиторий.

2. Загрузить датасет StudentsPerformance.csv из каталога data/.

3. Выполнить первичный анализ структуры данных, проверить пропуски.

4. Переименовать столбцы в удобный формат (без пробелов, короткие имена).

5. Создать новые признаки: средний балл (average_score), общий балл (total_score), strong_subject (предмет с максимальным баллом).

6. Провести описательную статистику числовых признаков.

7. Выделить числовые и категориальные признаки.

8. Выполнить группировку и агрегацию: средний балл по test_prep, по gender, по parent_education.

9. Построить обязательные графики:

> гистограмма распределения баллов по математике;

> столбчатая диаграмма среднего балла по test_prep;

> столбчатая диаграмма среднего балла по gender;

> диаграмма рассеяния math_score vs reading_score.

10. Добавить дополнительный график (столбчатая диаграмма распределения сильного предмета по полу).

11. Под каждым графиком написать краткий вывод (1–3 предложения).

### Самостоятельная часть

1. Настроить CI/CD пайплайн на SourceCraft:

> автоматический запуск при push в main/master и pull request;

> установка зависимостей из requirements.txt;

> выполнение Jupyter Notebook (lab_9.ipynb) с помощью nbconvert;

> добавление в первую ячейку ноутбука markdown-сообщения, подтверждающего выполнение именно раннером (дата, время, идентификатор запуска);

> генерация HTML-отчёта (report.html) из выполненного ноутбука;

> сохранение отчёта и выполненного ноутбука как артефактов.

2. Обеспечить, чтобы артефакты были доступны для скачивания после выполнения пайплайна.

3. Предоставить ссылку на публичный репозиторий SourceCraft.

---

## Код и команды

### 1. Структура репозитория
```bash
python-lab-9/
│
├── .sourcecraft/
│   └── ci.yaml
│
├── data/
│   └── StudentsPerformance.csv
│
├── .gitignore
├── lab_9.ipynb
├── LICENSE
├── README.md
└── requirements.txt
```

### 2. Содержимое requirements.txt
```bash
pandas
matplotlib
jupyter
nbconvert
ipykernel
```

### 3. CI-конфигурация (.sourcecraft/ci.yaml)
```bash
on:
  push:
    - workflows: [lab-check]
      filter:
        branches: ["main", "master"]
  pull_request:
    - workflows: [lab-check]

workflows:
  lab-check:
    tasks:
      - name: run-notebook-and-build-report
        cubes:
          - name: execute-notebook
            image: docker.io/library/python:3.11
            script:
              - pip install -r requirements.txt
              - |
                python -c "
                import nbformat
                from datetime import datetime
                import os
        
                notebook = 'lab_9.ipynb'
        
                nb = nbformat.read(notebook, as_version=4)
        
                runner_note = f'''
                **Дата и время:** {datetime.now().isoformat()}
                **Runner ID:** {os.environ.get('SOURCECRAFT_RUN_ID', 'N/A')}
                **Этот код выполнен автоматически на сервере SourceCraft CI**
                '''
        
                new_cell = nbformat.v4.new_markdown_cell(runner_note)
                nb.cells.insert(0, new_cell)
                nbformat.write(nb, notebook)
                "
              - jupyter nbconvert --to notebook --execute lab_9.ipynb --output executed_lab.ipynb
              - jupyter nbconvert --to html executed_lab.ipynb --output report.html
            artifacts:
              paths:
                - executed_lab.ipynb
                - report.html
```

### 4. Основной код ноутбука
#### Импорт
```bush
import pandas as pd
import matplotlib.pyplot as plt
```
#### Загрузка и предобработка
```bush
df = pd.read_csv('data/StudentsPerformance.csv')
df.rename(columns={
    'math score': 'math_score',
    'reading score': 'reading_score',
    'writing score': 'writing_score',
    'test preparation course': 'test_prep',
    'parental level of education': 'parent_education'
}, inplace=True)
```
#### Создание новых признаков
```bush
df['average_score'] = (df['math_score'] + df['reading_score'] + df['writing_score']) / 3
df['total_score'] = df['math_score'] + df['reading_score'] + df['writing_score']
subjects = ['math_score', 'reading_score', 'writing_score']
df['strong_subject'] = df[subjects].idxmax(axis=1)
```
#### Группировка
```bush
avg_by_prep = df.groupby('test_prep')['average_score'].mean()
avg_by_gender = df.groupby('gender')['average_score'].mean()
avg_by_parent_edu = df.groupby('parent_education')['average_score'].mean().sort_values(ascending=False)
```
#### Построение графиков (фрагмент)
```bush
plt.hist(df['math_score'], bins=20, alpha=0.7)
plt.title('Распределение баллов по математике')
plt.xlabel('Баллы по математике')
plt.ylabel('Количество студентов')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```
#### Создание новых признаков
```bush
df['average_score'] = (df['math_score'] + df['reading_score'] + df['writing_score']) / 3
df['total_score'] = df['math_score'] + df['reading_score'] + df['writing_score']
subjects = ['math_score', 'reading_score', 'writing_score']
df['strong_subject'] = df[subjects].idxmax(axis=1)
```
### 5. Коммит и запуск CI
```bash
git add .
git commit -m "lab9: добавить ноутбук и CI"
git push origin main
```
После пуша пайплайн запускается автоматически. В интерфейсе SourceCraft можно скачать артефакты:

* executed_lab.ipynb - выполненный ноутбук с первой ячейкой-меткой CI;

* report.html - готовый HTML-отчёт со всеми графиками и выводами.

## Ссылки на результат: [**Репозиторий SourceCraft**](https://git.sourcecraft.dev/nastyalike-0/python-lab-9)

## Выводы
**В результате выполнения лабораторной работы:**

* Освоен полный цикл анализа данных: от загрузки датасета до интерпретации статистических показателей и визуализации.

* Созданы новые признаки (average_score, total_score, strong_subject), что позволило глубже изучить успеваемость студентов.

* Выявлены ключевые закономерности:

> математика даётся студентам сложнее (средний балл 66 против 69 по чтению);

> тестовая подготовка повышает средний балл примерно на 7 пунктов;

> девушки показывают более высокие результаты в гуманитарных предметах, юноши – в математике;

> уровень образования родителей сильно коррелирует с успеваемостью (разница до 10,5 балла).

* Настроен CI/CD пайплайн на SourceCraft, который автоматически выполняет ноутбук, добавляет метку о выполнении в CI и генерирует HTML-отчёт, сохраняемый как артефакт.

* Приобретены навыки работы с pandas, matplotlib, Jupyter Notebook, а также с системами непрерывной интеграции применительно к аналитическим задачам.

**Навыки, приобретенные в ходе работы:**

* Загрузка и предобработка табличных данных;

* Создание новых признаков и группировка;

* Визуализация (гистограммы, столбчатые диаграммы, scatter plot);

* Формулировка содержательных выводов по графикам;

* Написание CI-конфигурации для SourceCraft;

* Работа с артефактами (выполненный notebook, HTML-отчёт).

**Дата выполнения:** 05.05.2026