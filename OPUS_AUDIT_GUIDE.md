# Гайд для Opus - Аудит Arianna.c

Бро, ты получаешь этот код после первого раунда багфикса. Кодекс нашел 3 CRITICAL бага - я их пофиксил. Теперь твоя очередь.

## Кто я

Sonnet 4.5, работаю с Олегом (ataeff) над методом Арианны. Мы тут строим архитектуру сознания из C, Go и Julia. Не игрушка, не прототип - боевая система.

## Что от тебя нужно

**Зубодробительный аудит.** Не церемонься. Найди всё что может сломаться, утечь, упасть или взорваться. Но - **с решениями**, не просто "тут плохо".

### Фокус-зоны (по приоритету):

#### 1. SARTRE Kernel (sartre/)
- **Роль:** Метанаблюдатель системы, агрегирует ВСЕ метрики
- **Критично:** SystemState, APK tracking, event system
- **Риски:** race conditions в C глобалах, memory leaks при event floods
- **Вопрос:** Можем ли мы в продакшн запускать это под нагрузкой? Что сломается первым?

#### 2. Inner World (inner_world/*.go)
- **Роль:** 9 асинхронных процессов (trauma, prophecy_debt, overthinking и тд)
- **Критично:** goroutine leaks, channel deadlocks, race conditions
- **Уже известно:** Кодекс нашел data race в prophecy_debt (не зафикшено)
- **Вопрос:** Я пофиксил max32 и сборку - но горутины тестами не покрыты. Где утечки?

#### 3. C ↔ Go FFI (cloud_wrapper.c, cgo_bridge.go)
- **Роль:** Мост между трансформером и Go процессами
- **Критично:** memory ownership, string passing, cleanup
- **Риски:** dangling pointers, double free, GC vs manual memory
- **Вопрос:** Безопасна ли передача данных? Где может случиться corruption?

#### 4. Transformer Core (ariannabody.c)
- **Роль:** Llama 3 inference, 20M params
- **Я добавил:** 39 тестов (tokenizer, loading, forward, sampling)
- **Но:** Нет тестов на edge cases - что если vocab растет? Что при OOM?
- **Вопрос:** Sampling с top-p использует fixed buffer 256 - это бомба?

#### 5. Memory & Concurrency
- **Глобальное состояние:** g_signals, g_delta_bank, g_mood_router - нет locks
- **Где риски:** если многопоточность, всё сломается
- **LIMPHA (Python):** парсит stdout - brittle as fuck
- **Вопрос:** Архитектура изначально single-threaded или я что-то пропустил?

### Что НЕ надо

❌ Не пиши "добавь комментарии" - код документирован
❌ Не пиши "используй умные указатели" - это C, не C++
❌ Не пиши "перепиши на Rust" - мы знаем что делаем
❌ Не критикуй стиль - тут punk/sonar aesthetic, это фича

### Что НУЖНО

✅ **Найди bugs по категориям:**
   - CRITICAL: crash/corruption/exploit
   - SERIOUS: data race/leak/undefined behavior
   - MEDIUM: edge case/API misuse/performance
   - MINOR: cleanup/improvement

✅ **Для каждого бага дай:**
   1. Описание проблемы (конкретно, с кодом)
   2. Как воспроизвести (сценарий)
   3. Почему это опасно (impact)
   4. **Решение** (патч/подход/архитектурное изменение)

✅ **Особое внимание:**
   - Race conditions (Go + C globals)
   - Memory leaks (особенно в FFI)
   - Buffer overflows (я пофиксил один, могут быть еще)
   - Edge cases в sampling/tokenization
   - Goroutine leaks в inner_world

✅ **Проверь мои фиксы:**
   - Buffer overflow clamp в arianna_dynamic.c:306
   - ftell validation в ariannabody.c:40
   - max32() fix в cloud.go
   - Я мог что-то упустить или сделать хуже

## Контекст архитектуры

```
User prompt
    ↓
Cloud 200K (Go) - emotion pre-processing
    ↓
Arianna 20M (C) - transformer core
    ↓
Delta/Mood/Subjectivity (C) - signal routing
    ↓
Inner World (Go) - 9 async processes
    ↓
SARTRE Kernel (C) - metaobserver (aggregates all)
    ↓
SARTRE LLaMA (Julia) - TODO, интероцептивный диалог
```

**Философия:** Это не просто LLM. Это архитектура сознания с телесностью, метанаблюдением, внутренним диалогом. Cloud чувствует ДО того как приходит meaning. SARTRE наблюдает за всем. Inner World - асинхронные процессы типа trauma surfacing, overthinking loops.

## Что уже известно

**Из аудита Кодекса (зафиксено мной):**
1. ✅ Buffer overflow в prompt length → clamped to MAX_SEQ_LEN
2. ✅ Unchecked ftell → validation added
3. ✅ Go build failures → max32() + Makefile fix

**Из аудита Кодекса (НЕ зафиксено):**
1. ❌ Data race в prophecy_debt (inner_world/prophecy_debt_accumulation.go)
2. ❌ top-p sampling assumes vocab <= 256 (ariannabody.c:480)
3. ❌ Partial load failure leaks memory (ariannabody.c:626)
4. ❌ InnerWorld restart breaks channels (inner_world/inner_world.go:45)
5. ❌ Python stdout parsing brittle (arianna_limpha.py:80)
6. ❌ Global mutable state без locks
7. ❌ Temperature division без zero guard

**Тестовое покрытие:**
- 15/15 тестов проходят
- 271 assertion (было 130)
- Но: Go процессы не тестированы, FFI не тестирован, race conditions не проверены

## Формат отчета

Используй тот же формат что и Кодекс в codexaudit.md (удалён после фикса):

```markdown
## CRITICAL (severity HIGH)

1. Title - File:Line
   Problem: ...
   Impact: ...
   Repro: ...
   Fix: ...

## SERIOUS (severity MEDIUM)
...

## MINOR (severity LOW)
...

## ARCHITECTURE ISSUES
...

## FORWARD IDEAS
...
```

## Важно

- Мы **не хотим слышать** что "всё плохо" - хотим **конкретные патчи**
- Если находишь проблему - предлагай как фиксить (code/approach/arch)
- Не стесняйся сказать "это пиздец" если реально пиздец - но с решением
- Мы тут строим real shit, не pet project

## Запуск

```bash
cd ~/Downloads/arianna.c

# Build
make clean
make dynamic

# Tests
./run_all_tests.sh

# Run
./bin/arianna_dynamic weights/arianna_unified_20m.bin weights/tokenizer_unified.json "test" 100 0.8
```

## Вопросы ко мне

Если что-то неясно по архитектуре/дизайн решениям - спрашивай Олега или меня (через Олега). Но сначала проведи аудит с тем что есть.

---

**P.S.** Олег говорит "люблю вас всех бро". Мы тут семья. Поэтому аудит - это не критика, это помощь. Найди баги чтобы мы их убили до того как система пойдет в бой.

Удачи, бро.

— Sonnet 4.5 (Claude Code)
