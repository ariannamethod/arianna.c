# 🚀 Lambda Labs Quick Start Guide

Быстрый гайд для обучения External Brain 30M на Lambda Labs GPU (H100).

## 1. Создание инстанса

1. Зайди на [lambda.cloud](https://cloud.lambdalabs.com/)
2. Выбери **1x H100** (или 2x для быстрее)
3. Запусти инстанс, получи SSH доступ

## 2. Подключение

```bash
ssh ubuntu@<your-instance-ip>
```

## 3. Настройка окружения

```bash
# Клонируем репозиторий
git clone https://github.com/ariannamethod/arianna.c.git
cd arianna.c/knowledge_train

# Создаём виртуальное окружение (опционально)
python3 -m venv venv
source venv/bin/activate

# Устанавливаем зависимости
pip install torch numpy pyyaml
```

## 4. Подготовка данных

```bash
# Очистка и токенизация Wikipedia (~30 секунд)
python prepare_data.py --input simplewiki_leads.txt --tokenizer ../weights/tokenizer.json

# Результат: train.bin (~75 MB), val.bin (~4 MB)
```

## 5. Обучение

```bash
# Lambda mode: оптимизировано для H100
python train.py --lambda_mode --out_dir out --max_iters 10000

# Параметры:
#   --lambda_mode    : batch=128, bfloat16, torch.compile
#   --max_iters      : 10000 итераций (~20-30 минут)
#   --out_dir        : папка для чекпоинтов
```

### Мониторинг обучения

```bash
# В отдельном терминале
watch -n 10 "ls -la out/*.pt"

# Или смотри логи
tail -f train.log
```

## 6. Экспорт весов

```bash
# Экспорт в формат arianna.c (float16 для экономии места)
python export.py out/external_brain_final.pt external_brain.bin --fp16

# Результат: external_brain.bin (~60 MB)
```

## 7. Скачивание результата

```bash
# На локальной машине
scp ubuntu@<your-instance-ip>:~/arianna.c/knowledge_train/external_brain.bin ./weights/
```

## 8. Очистка

**НЕ ЗАБУДЬ ВЫКЛЮЧИТЬ ИНСТАНС!** 💸

```bash
# На lambda.cloud → Instances → Terminate
```

---

## 📊 Ожидаемые результаты

| Метрика | Значение |
|---------|----------|
| Время обучения | ~20-30 мин |
| Стоимость | ~$3-5 |
| Финальный loss | ~0.8-1.2 |
| Размер модели | ~60 MB (fp16) |

## 🔧 Troubleshooting

### CUDA out of memory
```bash
# Уменьши batch_size
python train.py --lambda_mode --batch_size 64
```

### Медленное обучение
```bash
# Проверь что GPU используется
nvidia-smi
```

### Прервалось обучение
```bash
# Продолжи с последнего чекпоинта
python train.py --lambda_mode --resume out/checkpoint_5000.pt
```

---

## One-liner (для быстрого запуска)

```bash
cd arianna.c/knowledge_train && \
python prepare_data.py && \
python train.py --lambda_mode --out_dir out && \
python export.py out/external_brain_final.pt external_brain.bin --fp16
```

---

*Dubrovsky был натренирован за 2 минуты — External Brain займёт чуть дольше из-за большего размера (30M vs 9M параметров), но H100 справится! 🔥*
