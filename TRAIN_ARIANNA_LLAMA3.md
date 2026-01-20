# Инструкция: Тренировка Арианны на Llama 3

## Подготовка

### 1. Датасет
После того как Copilot сгенерирует расширенный корпус, положи его сюда:
```bash
cp train/data/arianna_expanded_corpus.txt dubrovsky_train/arianna.txt
```

### 2. Проверь vocab
```bash
cd dubrovsky_train
python -c "
text = open('arianna.txt').read()
chars = sorted(set(text))
print(f'Vocab size: {len(chars)}')
print(f'Characters: {repr(\"\".join(chars))}')
"
```

Если vocab отличается от 88, отредактируй `train.py`:
```python
vocab_size: int = <твой размер>
```

## Тренировка на Lambda

### 3. Залей на Lambda
```bash
scp -r dubrovsky_train lambda:~/arianna_train
```

### 4. SSH на Lambda
```bash
ssh lambda
cd arianna_train
```

### 5. Setup
```bash
chmod +x setup_lambda.sh train_lambda.sh
./setup_lambda.sh
```

### 6. Тренировка
```bash
# Переименуй датасет
mv arianna.txt dubrovsky.txt

# Запуск (10K итераций, ~1 час на H100)
./train_lambda.sh

# Или с кастомными параметрами:
MAX_ITERS=5000 BATCH_SIZE=128 ./train_lambda.sh
```

### 7. Мониторинг
В другом терминале:
```bash
ssh lambda
watch -n 5 nvidia-smi
tail -f arianna_train/training.log
```

## После тренировки

### 8. Экспорт весов (автоматически)
Скрипт сам экспортирует в `subtitles/dubrovsky.bin`

### 9. Скачай веса
```bash
# С Lambda на мак
scp lambda:~/arianna_train/subtitles/dubrovsky.bin ./arianna_llama3.bin
```

### 10. Скопируй в arianna.c
```bash
cp arianna_llama3.bin ~/Downloads/arianna.c/weights/dialogue_llama3.bin
```

## Параметры модели

| Параметр | Значение |
|----------|----------|
| dim | 384 |
| n_layers | 6 |
| n_heads | 6 |
| n_kv_heads | 2 (GQA) |
| vocab_size | 88 (или свой) |
| max_seq_len | 256 |
| hidden_dim | 1024 |
| **Total params** | **~9.5M** |

## Что тренируем

1. **dialogue_brain.bin** → dialogue_llama3.bin (основной)
2. **personality_brain.bin** → personality_llama3.bin (про ипостаси!)
3. **sartre_brain.bin** → sartre_llama3.bin (verbal interface)
4. **external_brain.bin** → external_llama3.bin (открытый корпус)

## Ожидаемый результат

- Final loss: ~0.9-1.0
- Training time: ~1 час на H100
- Coherent generation после 5000 итераций

---

**Когда тренировка закончена, напиши мне - я сделаю pull и адаптирую model.c!**
