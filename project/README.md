# IRF Kriging Forecasting Service

Сквозной ИИ-проект для прогнозирования месячных чисел солнечных пятен по реальному датасету SILSO Monthly Mean Total Sunspot Number V2.0. Проект включает подготовку данных, обучение модели IRF Kriging, сравнение с baseline-моделями, FastAPI-сервис, тесты и Docker-упаковку.

## 1. Краткое описание

Цель проекта — построить воспроизводимую ИИ-систему, которая прогнозирует значения одномерного временного ряда и возвращает не только точечный прогноз, но и оценку неопределённости прогноза.

В качестве основной модели используется `IRFKriging` с ядром:

```text
RationalQuadratic + Periodic(T=season_period) + Nugget
```

Основной датасет:

```text
project/data/raw/SN_m_tot_V2.0.csv
```

Источник данных: SILSO Monthly Mean Total Sunspot Number V2.0.

## 2. Структура проекта

```text
project/
  README.md
  report.md
  self-checklist.md

  Dockerfile
  docker-compose.yml
  requirements.txt
  .dockerignore

  artifacts/
    model.pkl
    metrics.json
    baseline_metrics.csv
    baseline_summary.json

  data/
    raw/
      SN_m_tot_V2.0.csv
    processed/
      silso_monthly_sunspots_1970.csv
      test_predictions.csv
      baseline_predictions.csv

  src/
    models/
      download_dataset.py
      train.py
      predict.py
      evaluate_baselines.py
      irf_kriging.py
      kernels.py

    service/
      __init__.py
      main.py
      schemas.py
      model_loader.py

  tests/
    test_model.py
    test_api.py
```

## 3. Установка зависимостей

Рекомендуется использовать виртуальное окружение.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r project/requirements.txt
```

Проверка установки:

```powershell
python -m pytest project/tests
```

## 4. Данные

Проект использует локальный CSV-файл:

```text
project/data/raw/SN_m_tot_V2.0.csv
```

Если файл ещё не скачан, его можно скачать отдельным скриптом:

```powershell
python project/src/models/download_dataset.py
```

Если на Windows возникает SSL-ошибка при скачивании, можно скачать файл вручную из источника SILSO и положить его в `project/data/raw/` под именем `SN_m_tot_V2.0.csv`.

В обучении используются записи начиная с 1970 года. Значения `SSn < 0` отбрасываются как пропущенные/некорректные.

## 5. Обучение модели

Базовый запуск:

```powershell
python project/src/models/train.py
```

Быстрый запуск для проверки:

```powershell
python project/src/models/train.py --max-train-points 120 --optim-type DE
```

Запуск с параметрами, использованными в текущем эксперименте:

```powershell
python project/src/models/train.py --max-train-points 180 --optim-type grad --season-period 132
```

После обучения создаются артефакты:

```text
project/artifacts/model.pkl
project/artifacts/metrics.json
project/data/processed/silso_monthly_sunspots_1970.csv
project/data/processed/test_predictions.csv
```

Текущие метрики финальной модели:

| Метрика | Значение |
|---|---:|
| MAE | 21.735 |
| RMSE | 27.244 |
| MAPE, % | 653.638 |
| sMAPE, % | 57.644 |
| Train used | 180 |
| Test | 136 |

`MAPE` для этого датасета нестабилен, потому что ряд солнечных пятен может содержать значения, близкие к нулю. Поэтому для анализа качества важнее смотреть на `MAE`, `RMSE`, `sMAPE` и график прогноза.

## 6. Прогноз из командной строки

Прогноз на 24 месяца вперёд:

```powershell
python project/src/models/predict.py --horizon 24
```

Прогноз по заданной сетке:

```powershell
python project/src/models/predict.py --grid 540 541 542 543 544
```

Пример ответа:

```json
{
  "grid": [540.0, 541.0, 542.0],
  "predicted_mean": [101.67, 98.80, 95.49],
  "predicted_variance": [181.74, 283.81, 394.91]
}
```

## 7. Baseline-сравнение

Сравнение с простыми моделями:

```powershell
python project/src/models/evaluate_baselines.py
```

Результаты сохраняются в:

```text
project/artifacts/baseline_metrics.csv
project/artifacts/baseline_summary.json
project/data/processed/baseline_predictions.csv
```

Текущая таблица сравнения:

| Model | MAE | RMSE | MAPE, % | sMAPE, % |
|---|---:|---:|---:|---:|
| IRFKriging | 21.735 | 27.244 | 653.638 | 57.644 |
| SeasonalNaive | 19.477 | 29.096 | 94.524 | 50.378 |
| MovingAverage | 62.592 | 71.978 | 2326.746 | 91.607 |
| NaiveLast | 64.342 | 74.228 | 2399.110 | 92.225 |
| PolynomialRegression | 537.799 | 609.039 | 10370.214 | 162.220 |

Вывод: IRF Kriging даёт лучший `RMSE` среди рассмотренных моделей, хотя по `MAE` и `sMAPE` близкий конкурент — seasonal naive baseline. Это ожидаемо для солнечной активности, где сильна цикличность около 11 лет.

## 8. FastAPI-сервис

Запуск API локально:

```powershell
python -m uvicorn project.src.service.main:app --reload
```

После запуска:

```text
http://127.0.0.1:8000/docs
```

Endpoints:

| Method | Endpoint | Назначение |
|---|---|---|
| GET | `/health` | Проверка состояния сервиса и наличия модели |
| GET | `/model/info` | Информация о модели и метриках |
| POST | `/predict` | Прогноз по horizon или grid |
| GET | `/metrics` | Простые технические метрики сервиса |

Пример запроса на прогноз:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d "{\"horizon\": 24}"
```

Пример запроса по конкретной сетке:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d "{\"grid\": [540, 541, 542, 543, 544]}"
```

## 9. Тесты

Запуск всех тестов:

```powershell
python -m pytest project/tests
```

На текущей версии проекта тесты проходят:

```text
7 passed
```

Проверяются:

- наличие `model.pkl`;
- загрузка модели;
- корректность `predict(grid)`;
- endpoint `/health`;
- endpoint `/model/info`;
- endpoint `/predict`;
- отклонение некорректного входа.

## 10. Docker

Сборка образа из корня репозитория:

```powershell
docker build -t irf-kriging-api project
```

Запуск контейнера:

```powershell
docker run --rm -p 8000:8000 irf-kriging-api
```

Проверка:

```powershell
curl http://127.0.0.1:8000/health
```

Альтернатива через compose:

```powershell
docker compose -f project/docker-compose.yml up --build
```

## 11. Наблюдаемость

В сервис добавлены:

- endpoint `/health`;
- endpoint `/metrics`;
- счётчик запросов;
- счётчик ошибок;
- средняя latency запросов;
- проверка наличия модели.

## 12. Ограничения проекта

- Модель обучается на одном временном ряде SILSO.
- `MAPE` плохо интерпретируется из-за значений ряда, близких к нулю.
- API использует заранее обученный `model.pkl`; online-обучение через API не реализовано.
- Текущая наблюдаемость базовая и не подключена к Prometheus/Grafana.
- Нет полноценного MLflow-трекинга экспериментов, результаты сохраняются в CSV/JSON.

## 13. Возможные улучшения

- Добавить графики `actual vs forecast` и доверительных интервалов.
- Добавить MLflow для трекинга экспериментов.
- Добавить endpoint `/fit_predict` для обучения на пользовательских данных.
- Добавить Prometheus/Grafana.
- Подобрать гиперпараметры модели на более широком наборе экспериментов.
- Добавить больше baseline-моделей для временных рядов.
