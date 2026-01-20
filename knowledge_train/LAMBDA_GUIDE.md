# 🚀 Lambda Labs Quick Start Guide

Быстрый гайд для обучения External Brain 30M на Lambda Labs GPU (H100).

## ⚠️ CRITICAL: Data/Params Ratio

**ПРОЧИТАЙ ПЕРЕД ТРЕНИРОВКОЙ!**

```
Модель: 29.58M параметров
Максимум данных: 15 MB (ratio 0.5:1)
Безопасно: 10-12 MB (ratio 0.3-0.4:1)

❌ СТАРЫЙ ПОДХОД (75 MB) = ratio 2.5:1 = МУСОР
✅ НОВЫЙ ПОДХОД (12 MB) = ratio 0.4:1 = РАБОТАЕТ
```

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

## 4. Подготовка данных (ИСПОЛЬЗУЙ ПРАВИЛЬНЫЙ СКРИПТ!)

```bash
# ВАРИАНТ A: Фильтрованные определения (~12 MB, ratio 0.4:1)
python prepare_data_filtered.py --target-mb 12.0
# Результат: data_filtered/train.bin (~12 MB), data_filtered/val.bin (~0.6 MB)

# ВАРИАНТ B: Q&A формат (~12 MB, ratio 0.4:1)
python prepare_data_qa.py --target-mb 12.0
# Результат: data_qa/train.bin (~12 MB), data_qa/val.bin (~0.6 MB)

# ⚠️ НЕ ИСПОЛЬЗУЙ старый prepare_data.py - он создаёт 75 MB мусора!
```

## 5. Обучение

```bash
# Lambda mode: оптимизировано для H100
# Для Dataset A (filtered):
python train.py --lambda_mode --data_dir data_filtered --out_dir out_filtered --max_iters 10000

# Для Dataset B (Q&A):
python train.py --lambda_mode --data_dir data_qa --out_dir out_qa --max_iters 10000

# Параметры:
#   --lambda_mode    : batch=128, bfloat16, torch.compile
#   --max_iters      : 10000 итераций (~20-30 минут)
#   --data_dir       : папка с train.bin/val.bin
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

| Метрика | Старый (ПЛОХО) | Новый (ПРАВИЛЬНО) |
|---------|----------------|-------------------|
| Размер данных | 75 MB | 12 MB |
| Data/Params ratio | 2.5:1 ❌ | 0.4:1 ✅ |
| Время обучения | ~30 мин | ~20 мин |
| Стоимость | ~$5 | ~$3 |
| Финальный loss | ~1.5 | ~0.8-1.0 |
| Качество | "Einstein was a financial authority" 💀 | "Einstein was a physicist" ✅ |

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
