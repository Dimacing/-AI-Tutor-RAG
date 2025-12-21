# AI Tutor RAG MVP

<img width="2134" height="1210" alt="image" src="https://github.com/user-attachments/assets/033c2559-684a-47b5-a964-a075653656e1" />

Минимальный MVP AI-репетитора на базе RAG (Retrieval-Augmented Generation).
Система индексирует материалы, находит релевантные фрагменты и генерирует ответ с источниками, самопроверкой и тестом.

## Возможности
- Полный RAG-пайплайн: ingest -> chunk -> embed -> index -> retrieve -> prompt -> answer
- Ответы с источниками, самопроверкой и тестами
- Режимы: документация Cloud.ru, локальные файлы (PDF/DOCX/TXT), сайт
- Интерфейсы: веб-UI (в стиле Cloud.ru), Streamlit, Telegram-бот
- Базовые проверки безопасности (profanity + PII), логирование, health/metrics endpoints
- Выбор LLM-провайдера и модели, выбор модели эмбеддингов, гиперпараметры поиска

Важно: LLM обязателен. При недоступности провайдера API вернет 503.

## Быстрый старт (локально)
1) Создайте виртуальное окружение и установите зависимости:
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2) Скопируйте файл окружения и настройте параметры:
```
copy .env.example .env
```
Укажите ключи LLM и нужные значения в `.env`.

3) Запустите бэкенд:
```
uvicorn backend.main:app --reload
```

4) Откройте веб-интерфейс:
- `http://localhost:8000` (или `http://localhost:8000/site`)

5) (Опционально) Запустите Streamlit:
```
streamlit run ui/streamlit_app.py
```

6) (Опционально) Запустите Telegram-бота:
```
python -m backend.telegram_bot
```

## Запуск в Docker
1) Установите Docker и Docker Compose.

2) Скопируйте файл окружения и укажите ключи:
```
copy .env.example .env
```

3) Запуск API:
```
docker compose up --build
```

4) Откройте `http://localhost:8000`.

5) Streamlit (опционально):
```
docker compose --profile streamlit up --build
```

6) Telegram-бот (опционально):
```
docker compose --profile bot up --build
```

Примечания:
- Индексы и загрузки сохраняются в `./data` и монтируются в контейнер.
- Если используете Ollama на хосте, укажите:
  - Windows/Mac: `OLLAMA_BASE_URL=http://host.docker.internal:11434`
  - Linux: добавьте host-gateway или используйте IP хоста.

## Режимы работы
### Документация Cloud.ru
Нажмите «Создать индекс» (или «Build cloud index») для переиндексации данных из `data/sources.json`.
Индекс хранится в `data/index`.

### Мои файлы
Загрузите PDF/DOCX/TXT и создайте локальный индекс.
Индекс хранится в `data/index_local`.

### Сайт
Вставьте URL, при необходимости включите обход ссылок и создайте индекс.
Индекс хранится в `data/index_web`.

## Выбор модели
Провайдера и модель можно выбрать в UI. Настройки - в `.env`:
- OpenAI: `OPENAI_API_KEY`, `OPENAI_MODEL`
- Gemini: `GEMINI_API_KEY`, `GEMINI_MODEL`
- DeepSeek: `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, `DEEPSEEK_BASE_URL`
- Ollama: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`
- GigaChat: `GIGACHAT_AUTH_KEY` (рекомендуется) или `GIGACHAT_ACCESS_TOKEN`,
  `GIGACHAT_MODEL`, `GIGACHAT_BASE_URL`
  - Для корпоративных сертификатов: `GIGACHAT_CA_BUNDLE`, `GIGACHAT_VERIFY_SSL`

## Индексация через CLI
```
python -m backend.rag.index --sources data/sources.json
```
Параметры: `--chunk-size`, `--overlap`, `--allowlist`, `--max-pages`.

## Гиперпараметры
В веб-UI доступны параметры:
- `chunk_size` и `overlap` для индексации
- `top_k` и `min_score` для поиска

## Пример запроса
«Я прочитал про Object Storage (S3), но не понимаю отличие от блочного хранилища. Объясни просто и задай вопрос для самопроверки.»

## Примечания по RAG
- Документы режутся на чанки с перекрытием.
- Эмбеддинги строятся с `sentence-transformers` и сохраняются в `data/index*`.
- Поиск - по косинусному сходству, низкие скоры отфильтровываются.
- Промпт требует цитировать источники и формировать самопроверку.

## Безопасность и данные
- Входные запросы проходят базовые проверки profanity + PII.
- Логи включены; `LOG_USER_TEXT=false` отключает запись текста пользователя.
- Пользовательские данные не сохраняются, кроме логов.

## Документы
- Архитектура: `docs/architecture.md`
- Пошаговая инструкция: `docs/step-by-step.md`
- Demo-скрипт: `docs/demo-script.md`
- План презентации: `docs/presentation-outline.md`
- Итоговый отчет: `docs/final-report.md`

## Структура проекта
```
backend/          FastAPI сервис и RAG пайплайн
ui/               Streamlit UI
scripts/          Вспомогательные скрипты
data/             Источники и индексы
/docs             Архитектура и демо-материалы
```
