# Лабораторная работа №3: CI/CD для статического сайта

## Цель работы

Изучить процесс настройки непрерывной интеграции и доставки (CI/CD) для статического сайта на платформах SourceCraft и GitHub. В рамках работы необходимо настроить автоматическую сборку и публикацию сайта, сгенерированного с помощью MkDocs, при каждом обновлении кода в репозитории.

## Задания

1. Авторизоваться в SourceCraft через аккаунт Яндекс.
2. Создать публичную организацию и пустой репозиторий.
3. Сгенерировать персональный токен доступа (PAT) с правами Maintainer.
4. Добавить удалённый репозиторий SourceCraft в локальный проект.
5. Настроить CI/CD пайплайн с использованием встроенных инструментов SourceCraft.
6. Настроить CI/CD пайплайн на GitHub с использованием GitHub Actions.
7. Предоставить ссылки на репозитории и сформированные статические сайты.

## Код и команды

### Настройка окружения для SourceCraft

Для работы потребуется локальный репозиторий с проектом MkDocs. После создания репозитория в SourceCraft выполняется добавление второго удалённого источника:

```bash
git remote add sourcecraft https://<имя_аккаунта>:<персональный_токен>@git.sourcecraft.dev/<имя_аккаунта>/<имя_репозитория>.git
```

Проверка добавления:

```bash
git remote -v
```

### Структура проекта

```
main/
├── .source/          # Исходные файлы MkDocs
│   ├── docs/              # Markdown файлы
│   │    └── ... 
│   └── mkdocs.yml         # Конфигурация MkDocs
├── docs/                           # папка для собранного сайта (создаётся автоматически)
│   └── ...               
├── .sourcecraft/
│   ├── ci.yaml            # Конфигурация CI/CD пайплайна для SourceCraft
│   └── sites.yaml         # Настройка публикации сайта в SourceCraft
├── .github/
│   └── workflows/
│       └── deploy.yml     # Конфигурация CI/CD пайплайна для GitHub
├── requirements.txt       # Зависимости (mkdocs, mkdocs-material)
└── README.md
```

### 1. Настройка CI/CD для SourceCraft

#### Конфигурация CI/CD (ci.yaml)

Файл `.sourcecraft/ci.yaml` определяет пайплайн, который запускается при пуше в ветку `main`. В нём описаны две задачи: сборка сайта и публикация в ветку release.

```yaml
on:
  push:
    workflows: build-site
    filter:
      branches: main

workflows:
  build-site:
    tasks:
      - name: build-and-publish-site
        cubes:
          - name: build-mkdocs-site
            image: docker.io/library/python:3.13-slim
            script:
              - python -m pip install --upgrade pip
              - "if [ -f requirements.txt ]; then pip install -r requirements.txt; fi"
              - cd source && mkdocs build -d ../docs
              - echo "Сайт собран в папке ./docs"
              - ls -la ./docs

          - name: publish-to-release-branch
            script:
              - git checkout -b release
              - git add ./docs
              - "git commit -m \"feat: автоматическое обновление сайта\""
              - "git push origin release -f"
```
**Описание шагов:**

* build-mkdocs-site: в контейнере Python устанавливаются зависимости из requirements.txt, после чего в папке source выполняется команда mkdocs build -d ../docs. Собранный статический сайт оказывается в каталоге docs на уровне корня репозитория.


* publish-to-release-branch: создаётся (или переключается) ветка release, в неё добавляется папка docs и изменения принудительно пушатся в удалённый репозиторий. Ветка release используется для хранения скомпилированной версии сайта.

#### Настройка публикации (sites.yaml)

Файл `sites.yaml` указывает SourceCraft, из какой папки и какой ветки развёртывать сайт:

```yaml
site:
  root: docs
  ref: release
```

#### Нюансы и особенности решения

* **Использование персонального токена.** Токен необходим для аутентификации при добавлении удалённого репозитория.

* **Публикация через ветку release.** Вместо того чтобы публиковать docs/ напрямую в ветку main, используется отдельная ветка release. Это даёт возможность хранить историю исходников и собранного сайта раздельно.

* **Принудительный пуш (-f).** Поскольку ветка release существует только для хранения последней версии сайта, принудительная перезапись позволяет поддерживать её в актуальном состоянии без лишних коммитов.

Проверка наличия requirements.txt
Скрипт сборки сначала проверяет наличие файла requirements.txt и устанавливает зависимости только при его существовании, что делает пайплайн более гибким.

#### Результат SourceCraft

- **Репозиторий:** [https://sourcecraft.dev/nastyalike-0/main](https://nastyalike-0.sourcecraft.site/main/)
- **Сайт:** [https://nastyalike-0.sourcecraft.site/main/](https://nastyalike-0.sourcecraft.site/main/)

### 2. Настройка CI/CD для GitHub

#### Конфигурация CI/CD (deploy.yml)

Файл `.github/workflows/deploy.yml` определяет пайплайн для GitHub Actions:

```yaml
on:
  push:
    branches: [ main ]

permissions:
  contents: write

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: Install dependencies
        run: |
          pip install --upgrade pip
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

      - name: Build site
        run: |
          cd source                      
          mkdocs build -d ../docs       

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
          publish_branch: gh-pages
```

#### Настройка GitHub Pages

1. Перейти в **Settings** -> **Pages** репозитория.
2. В разделе **Branch** выбрать `gh-pages` и папку `/ (root)`.

#### Результат GitHub

- **Репозиторий:** [https://github.com/TwentyQ/twentyq.github.io/tree/gh-pages](https://nastyalike-0.sourcecraft.site/main/)
- **Сайт:** [https://twentyq.github.io/](https://nastyalike-0.sourcecraft.site/main/)

## Ключевых элементов

* runs-on: ubuntu-latest: Указывает операционную систему для выполнения workflow. 

* uses: actions/checkout@v4: Действие (action), которое скачивает код репозитория на виртуальную машину.

* uses: actions/setup-python@v5: Устанавливает указанную версию Python на виртуальную машину.

* secrets.GITHUB_TOKEN: Автоматически генерируемый токен, который GitHub Actions предоставляет для каждого workflow. Не требует ручного создания, в отличие от SourceCraft.

* peaceiris/actions-gh-pages@v3: Готовое действие для публикации статического сайта в ветку `gh-pages`.


## Выводы

В ходе выполнения лабораторной работы:

**SourceCraft:**

* Освоена настройка CI/CD на платформе SourceCraft с использованием встроенных пайплайнов.

* Реализована автоматическая сборка статического сайта на базе MkDocs.

* Настроена публикация собранного сайта в отдельную ветку `release`.

* Получен практический опыт работы с персональными токенами доступа.

**GitHub:**

* Изучена настройка GitHub Actions для автоматической сборки и публикации сайта.

* Освоено использование готовых действий (`actions/checkout`, `actions/setup-python`, `peaceiris/actions-gh-pages`).

* Настроена публикация на GitHub Pages через ветку `gh-pages`.

* Изучены ключевые концепции GitHub Actions (runs-on, permissions, secrets.GITHUB_TOKEN).

**Общее:**

* Создана единая структура проекта, работающая на обеих платформах.

* Настроена автоматическая публикация сайта при каждом пуше в ветку `main`.

* Получен опыт работы с различными подходами к CI/CD.

---

**Дата выполнения:** 25.03.2026