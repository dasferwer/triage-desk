# Запуск и обслуживание

## Обычный запуск

```bash
docker compose up --build -d --wait --wait-timeout 180
docker compose ps
curl -fsS http://localhost:8140/health
docker compose logs --tail=100 api dispatcher worker
```

API доступно на localhost:8140; PostgreSQL и RabbitMQ наружу не опубликованы.
Health API проверяет доступность базы и показывает возраст heartbeat
фоновых процессов. Docker отдельно проверяет dispatcher и worker.
`/metrics` содержит счётчики запросов и распределение задержек. Request ID
возвращается в заголовке и попадает в лог без текста обращения.

## Проверки

```bash
uv sync --extra dev --frozen
uv run ruff check .
uv run ruff format --check .
docker compose --profile test build test
docker compose --profile test run --rm test
docker compose exec -T api python scripts/smoke.py
uv run python scripts/recovery_smoke.py
docker compose --profile test stop test-db
```

Тестовая база `triagedesk_test` отдельная и использует tmpfs. Перед очисткой
фикстура требует TESTING и имя базы с `_test`. Recovery script запускается
с хоста из папки проекта: он временно останавливает только его RabbitMQ и worker.
Обычные запросы продолжают приниматься, но обработка может ждать восстановления.

## Новая версия модели

1. Выгрузить review-строки через `/feedback/export` под оператором. При наличии
   нескольких страниц объединить `rows` в один JSON-массив. Сохранить файл
   за пределами репозитория: это могут быть тексты пользователей.
2. Обучить новую версию:

```bash
uv run python scripts/train.py --version banking77-v2 --feedback /path/to/feedback.json
```

3. Изучить `models/banking77-v2/evaluation.json`, сравнить validation и test,
   проверить число принятых и исключённых исправлений. Повторные эксперименты
   по одному test постепенно превращают его в часть разработки; для дальнейшего
   подбора понадобится новый независимый контрольный набор.
4. Пересобрать сервисы и зарегистрировать локальный артефакт:

```bash
docker compose up --build -d --wait --wait-timeout 180
docker compose exec -T api python scripts/seed.py
```

5. Под оператором вызвать `POST /models/banking77-v2/activate`. Новые обращения
   получат v2. Старые сохранят v1. Для отката вызвать такой же endpoint с v1.

Seed не меняет активную модель при повторном запуске. Все версии, на которые
ссылаются обращения, нужно сохранять в образе. Заменять файлы существующей
версии нельзя: seed отклонит другой hash manifest.

## Если обработка остановилась

- Pending: проверить dispatcher и RabbitMQ.
- Queued: сообщение могло потеряться до подтверждения; dispatcher повторит его после lease.
- Processing: после падения worker задание снова станет доступно по окончании lease.
- Needs review с inference_unavailable: посмотреть тип ошибки worker и проверить
  наличие модели. Исправить тему через API оператора.

Не менять generation, attempts и статусы вручную в SQL. Повторные публикации
и истечение lease уже обработаны в сервисе.

`docker compose down` останавливает проект и сохраняет тома. `down --volumes`
удаляет его базу и очередь и не нужен для обычной остановки.
