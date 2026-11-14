# blood-cell-sep

Метод глубокого обучения для разделения слипшихся клеток крови на микроскопических изображениях.

## Идея метода

U-Net с двумя выходами:
1. маска «клетка / фон»;
2. карта расстояний до края клетки.

Пики карты расстояний служат маркерами центров клеток, затем watershed разрезает слипшиеся комки на отдельные клетки.
Для обучения дополнительно генерируются синтетические слипания.

## Структура

```
blood-cell-sep/
  src/          # основной код: данные, модель, лоссы, постобработка, метрики
  scripts/      # запускаемые скрипты: загрузка данных, обучение, оценка
  baselines/    # методы для сравнения: Otsu+watershed, Cellpose, StarDist
  configs/      # yaml-конфиги экспериментов
  notebooks/    # EDA и картинки для отчёта
  results/      # итоговые таблицы и рисунки
  data/         # данные (не в git, заполняется скриптами)
```

## Установка

Нужен Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

**PyTorch** (команда зависит от машины):

- Mac (Apple Silicon):
  ```bash
  pip install torch torchvision
  ```
- ПК с видеокартой NVIDIA: актуальную команду с нужной версией CUDA взять на
  https://pytorch.org/get-started/locally/ (вида `pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX`).

**Остальные зависимости:**

```bash
pip install -r requirements.txt
```

**Проверка:**

```bash
python scripts/check_env.py
```

На Mac должно показать `Устройство: mps`, на ПК — `Устройство: cuda` и название видеокарты.

## Рабочий процесс

- Код пишется на Mac → `git push`.
- Обучение на ПК: `git pull` → скачивание данных скриптом → запуск обучения.
- Данные и веса моделей в git не хранятся.
