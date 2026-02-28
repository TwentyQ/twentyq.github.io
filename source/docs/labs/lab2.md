# Лабораторная работа №2: Основы NumPy: массивы и векторные операции

## Цель работы
Изучить основы работы с библиотекой NumPy в Python: создание и обработка массивов, выполнение векторных и матричных операций, статистический анализ данных и визуализация результатов с использованием Matplotlib и Seaborn.

## Задание
1. Создать виртуальное окружение и установить необходимые зависимости (numpy, matplotlib, seaborn, pandas, pytest).

2. Реализовать функции для работы с массивами:

  * создание вектора от 0 до 9 

  * создание матрицы 5×5 со случайными числами
   
  * изменение формы массива

  * транспонирование матрицы

3. Реализовать векторные операции:

  * сложение векторов
   
  * умножение вектора на скаляр

  * поэлементное умножение

  * скалярное произведение

4. Реализовать матричные операции:

  * умножение матриц

  * вычисление определителя

  * нахождение обратной матрицы

  * решение системы линейных уравнений

5. Выполнить статистический анализ данных из CSV-файла:

  * загрузка данных

  * вычисление среднего, медианы, стандартного отклонения, минимума, максимума, перцентилей

  * нормализация данных

6. Построить графики:

  * гистограмма распределения оценок

  * тепловая карта корреляции предметов

  * линейный график оценок студентов

7. Обеспечить соответствие кода стандартам PEP-8

8. Добавить аннотации типов (PEP-484) для всех функций

9. Добавить документацию (PEP-257) для всех функций

10. Написать тесты для проверки корректности реализации

11. Оформить отчет и опубликовать его на статическом сайте

## Код и команды

1. Настройка окружения
```bash
# Создание виртуального окружения
python -m venv numpy_env

# Активация (Windows)
numpy_env\Scripts\activate

# Активация (Mac/Linux)
source numpy_env/bin/activate

# Установка зависимостей
pip install numpy matplotlib seaborn pandas pytest
```
2. Структура проекта
```text
numpy_lab/
├── main.py                 # Основные функции
├── test.py                 # Тесты
├── data/
│   └── students_scores.csv # Данные для анализа
└── plots/                  # Сохраненные графики
    ├── histogram.png
    ├── heatmap.png
    └── line_plot.png
```
3. Содержимое файла data/students_scores.csv
```csv
math,physics,informatics
78,81,90
85,89,88
92,94,95
70,75,72
88,84,91
95,99,98
60,65,70
73,70,68
84,86,85
90,93,92
```

4. Реализация функций в main.py
**Создание и обработка массивов**
```python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def create_vector() -> np.ndarray:
    """
    Создать массив от 0 до 9.
    
    Returns:
        numpy.ndarray: Массив чисел от 0 до 9 включительно
    """
    return np.arange(0, 10)


def create_matrix() -> np.ndarray:
    """
    Создать матрицу 5x5 со случайными числами [0,1].

    Returns:
        numpy.ndarray: Матрица 5x5 со случайными значениями от 0 до 1
    """
    return np.random.rand(5, 5)


def reshape_vector(vec: np.ndarray) -> np.ndarray:
    """
    Преобразование (10,) -> (2,5)
    
    Args:
        vec (numpy.ndarray): Входной массив формы (10,)
    
    Returns:
        numpy.ndarray: Преобразованный массив формы (2, 5)
    """
    return np.reshape(vec, (2, 5))


def transpose_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Транспонирование матрицы.

    Args:
        mat (numpy.ndarray): Входная матрица
    
    Returns:
        numpy.ndarray: Транспонированная матрица
    """
    return np.transpose(mat)
```
**Векторные операции**
```python
def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Сложение векторов одинаковой длины.
    
    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор
    
    Returns:
        numpy.ndarray: Результат поэлементного сложения
    """
    return a + b


def scalar_multiply(vec: np.ndarray, scalar: float | int) -> np.ndarray:
    """
    Умножение вектора на число.
    
    Args:
        vec (numpy.ndarray): Входной вектор
        scalar (float/int): Число для умножения
    
    Returns:
        numpy.ndarray: Результат умножения вектора на скаляр
    """
    return vec * scalar


def elementwise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Поэлементное умножение.
    
    Args:
        a (numpy.ndarray): Первый вектор/матрица
        b (numpy.ndarray): Второй вектор/матрица
    
    Returns:
        numpy.ndarray: Результат поэлементного умножения
    """
    return a * b


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Скалярное произведение.
    
    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор
    
    Returns:
        float: Скалярное произведение векторов
    """
    return np.dot(a, b)
```
**Матричные операции**
```python
def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Умножение матриц.

    Args:
        a (numpy.ndarray): Первая матрица
        b (numpy.ndarray): Вторая матрица
    
    Returns:
        numpy.ndarray: Результат умножения матриц
    """
    return np.matmul(a, b)


def matrix_determinant(a: np.ndarray) -> float:
    """
    Определитель матрицы.

    Args:
        a (numpy.ndarray): Квадратная матрица
    
    Returns:
        float: Определитель матрицы
    """
    return np.linalg.det(a)


def matrix_inverse(a: np.ndarray) -> np.ndarray:
    """
    Обратная матрица.

    Args:
        a (numpy.ndarray): Квадратная матрица
    
    Returns:
        numpy.ndarray: Обратная матрица
    """
    return np.linalg.inv(a)


def solve_linear_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решить систему Ax = b

    Args:
        a (numpy.ndarray): Матрица коэффициентов A
        b (numpy.ndarray): Вектор свободных членов b
    
    Returns:
        numpy.ndarray: Решение системы x
    """
    return np.linalg.solve(a, b)
```
**Статистический анализ**
```python
def load_dataset(path: str = "data/students_scores.csv") -> np.ndarray:
    """
    Загрузить CSV и вернуть NumPy массив.
    
    Args:
        path (str): Путь к CSV файлу
    
    Returns:
        numpy.ndarray: Загруженные данные в виде массива
    """
    return pd.read_csv(path).to_numpy()


def statistical_analysis(data: np.ndarray) -> dict[str, float]:
    """
    Статистический анализ данных.
    
    Args:
        data (numpy.ndarray): Одномерный массив данных
    
    Returns:
        dict: Словарь со статистическими показателями
    """
    stats = {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "25_percentile": np.percentile(data, 25),
        "75_percentile": np.percentile(data, 75)
    }
    return stats


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Min-Max нормализация: (x - min) / (max - min)

    Args:
        data (numpy.ndarray): Входной массив данных

    Returns:
        numpy.ndarray: Нормализованный массив данных в диапазоне [0, 1]
    """
    data_min: float = float(np.min(data))
    data_max: float = float(np.max(data))

    if data_max - data_min == 0:
        return np.zeros_like(data)

    return (data - data_min) / (data_max - data_min)
```
**Визуализация**
```python
def plot_histogram(data: np.ndarray) -> None:
    """
    Построить гистограмму распределения оценок по математике.

    Args:
        data (numpy.ndarray): Данные для гистограммы
    """
    plt.hist(data)
    plt.xlabel('Оценка')
    plt.ylabel('Частота')
    plt.title('Распределение оценок по математике')
    plt.savefig('plots/histogram.png')
    plt.close()


def plot_heatmap(matrix: np.ndarray) -> None:
    """
    Построить тепловую карту корреляции предметов.
    
    Args:
        matrix (numpy.ndarray): Матрица корреляции
    """
    sns.heatmap(matrix, annot=True, 
                xticklabels=['Математика', 'Физика', 'Информатика'],
                yticklabels=['Математика', 'Физика', 'Информатика'])
    plt.title('Матрица корреляции предметов')
    plt.savefig('plots/heatmap.png')
    plt.close()


def plot_line(x: np.ndarray, y: np.ndarray) -> None:
    """
    Построить график зависимости: студент -> оценка по математике.

    Args:
        x (numpy.ndarray): Номера студентов
        y (numpy.ndarray): Оценки студентов
    """
    plt.plot(x, y, marker='o')
    plt.title('Оценки студентов по математике')
    plt.xlabel('Номер студента')
    plt.ylabel('Оценка')
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/line_plot.png')
    plt.close()
```

5. Тестирование (test.py)
Тесты проверяют корректность работы всех функций. 

**Запуск тестов:**
```bash
python -m pytest test.py -v
```

**Пример одного из тестов:**
```python
def test_statistical_analysis() -> None:
    """
    Тест статистического анализа данных.
    """
    data: np.ndarray = np.array([10, 20, 30])
    result: dict[str, float] = statistical_analysis(data)
    assert result["mean"] == 20
    assert result["min"] == 10
    assert result["max"] == 30
    assert result["25_percentile"] == 15
    assert result["75_percentile"] == 25
```

**Результаты выполнения тестов:**
```bash
collected 17 items                                                                                                                                                                 

test.py::test_create_vector PASSED                                                                                                                                           [  5%]
test.py::test_create_matrix PASSED                                                                                                                                           [ 11%]
test.py::test_reshape_vector PASSED                                                                                                                                          [ 17%]
test.py::test_vector_add PASSED                                                                                                                                              [ 23%]
test.py::test_scalar_multiply PASSED                                                                                                                                         [ 29%]
test.py::test_elementwise_multiply PASSED                                                                                                                                    [ 35%]
test.py::test_dot_product PASSED                                                                                                                                             [ 41%]
test.py::test_matrix_multiply PASSED                                                                                                                                         [ 47%]
test.py::test_matrix_determinant PASSED                                                                                                                                      [ 52%]
test.py::test_matrix_inverse PASSED                                                                                                                                          [ 58%]
test.py::test_solve_linear_system PASSED                                                                                                                                     [ 64%]
test.py::test_load_dataset PASSED                                                                                                                                            [ 70%]
test.py::test_statistical_analysis PASSED                                                                                                                                    [ 76%]
test.py::test_normalization PASSED                                                                                                                                           [ 82%]
test.py::test_plot_histogram PASSED                                                                                                                                          [ 88%]
test.py::test_plot_heatmap PASSED                                                                                                                                            [ 94%]
test.py::test_plot_line PASSED                                                                                                                                               [100%]

=============================================================================== 17 passed in 6.79s ================================================================================
```
## Ссылки на результат
**Репозиторий:** https://github.com/TwentyQ/ITMO_PYTHON/tree/main/Lab_2_NumPy

## Выводы

**В результате выполнения лабораторной работы:**

* Освоены основы работы с библиотекой NumPy:

  * создание и манипуляция массивами различных форм

  * выполнение векторных и матричных операций

  * статистический анализ данных

* Изучены методы визуализации данных с использованием Matplotlib и Seaborn

* Применены стандарты оформления кода:

  * аннотации типов (PEP-484) для всех функций

  * документация (PEP-257)

  * соблюдение стиля кодирования (PEP-8)

* Реализовано тестирование функций с помощью pytest

* Интегрирован отчет в существующий сайт-портфолио на MkDocs

Дата выполнения: 01.03.2026