# LLM Text Pipeline

Учебный многошаговый пайплайн на Python: извлекает смысл обращения, определяет категорию и намерение, готовит сводку и три ключевых пункта, строит ответ по правилам категории и проверяет его на соответствие исходному тексту.

## Требования

- Python 3.10 или новее
- API-ключ OpenAI или совместимого провайдера
- Доступ к выбранной модели

Пайплайн делает пять последовательных вызовов модели на каждый текст. При временных ошибках API клиент повторяет запрос; если отдельный этап всё равно недоступен, применяется безопасный fallback и это отражается в результате.

## Создание и активация venv

Выполняйте команды из корня проекта.

### Linux и macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

После активации в начале строки терминала обычно отображается `(.venv)`. Чтобы выйти из окружения:

```bash
deactivate
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Если PowerShell запрещает запуск скрипта активации, используйте Command Prompt:

```bat
.venv\Scripts\activate.bat
```

## Настройка API-ключа

Скопируйте шаблон конфигурации:

```bash
cp .env.example .env
```

В Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Откройте `.env` и укажите `OPENAI_API_KEY`. При необходимости задайте модель в `LLM_MODEL`, адрес OpenAI-совместимого API — в `LLM_BASE_URL`, число повторов — в `LLM_MAX_RETRIES`. `.env` разбирается средствами стандартной библиотеки Python; файл исключён из Git.

## Запуск

Перед запуском активируйте `.venv`.

Передать текст прямо в командной строке:

```bash
python -m llm_pipeline.main --text "Не могу войти в аккаунт после смены пароля."
```

Прочитать текст из файла:

```bash
python -m llm_pipeline.main --file sample_input/support.txt
```

Запустить пять демонстрационных обращений:

```bash
python -m llm_pipeline.main --demo examples/demo_inputs.json
```

Получить структурированный JSON вместо удобного текстового вывода:

```bash
python -m llm_pipeline.main --text "Подскажите, как вернуть заказ?" --output json
```

Без `--text`, `--file` и `--demo` программа читает один текст из стандартного ввода:

```bash
python -m llm_pipeline.main
```

После установки `pip install -e .` также доступна команда `llm-pipeline` с теми же аргументами.

## Что возвращается

JSON содержит `summary`, `category`, `intent`, ровно три строки в `key_points`, `final_answer`, результат self-check, пояснение к проверке и признак использования fallback. Допустимые категории: `support`, `feedback`, `complaint`, `sales`, `general_question`.

## Структура проекта

```text
src/llm_pipeline/
  llm_client.py  # только вызов API, JSON и повтор временных ошибок
  prompts.py     # system prompt, шаблоны шагов и инструкции маршрутизации
  schemas.py     # модели и валидация Pydantic
  pipeline.py    # пять последовательных этапов и fallback
  main.py        # CLI
examples/        # набор из пяти демо-текстов
sample_input/    # отдельный пример входного текста
tasks/           # исходные задания по этапам и общее ТЗ
```

Подробные учебные задания собраны в [`tasks/`](tasks/).
