# Multi-Agent Educational System

Локальная multi-agent система для обучения студентов с использованием нескольких специализированных LLM-агентов.

Система поддерживает объяснение учебных понятий, генерацию заданий, проверку ответов, RAG, общую память между агентами и наблюдаемость через Langfuse.

---

## 1. Архитектура

Система состоит из следующих компонентов:

```text
                         Student
                            |
                            v
                    +---------------+
                    | Orchestrator  |
                    |    :5000      |
                    +-------+-------+
                            |
                            v
                         Router
                       /        \
                      /          \
                     v            v
              +-----------+  +-------------+
              |   Tutor   |  | Assessment  |
              |   :5001   |  |    :5002    |
              +-----+-----+  +------+------+
                    |               |
                    v               v
                   RAG            Tools
                    |               |
                    +-------+-------+
                            |
                            v
                     Shared Memory
                         SQLite
                            |
                            v
                         Ollama
                            |
                            v
                        Langfuse
```

---

## 2. Возможности

### Tutor Agent

Tutor используется для:

* объяснения математических и учебных понятий;
* простых примеров;
* проверки понимания;
* использования RAG;
* использования истории обучения студента.

### Assessment Agent

Assessment используется для:

* генерации учебных заданий;
* проверки ответов;
* анализа ошибок;
* выставления оценки;
* предоставления feedback;
* сохранения результатов обучения.

---

## 3. Технологии

| Компонент             | Технология              |
| --------------------- | ----------------------- |
| Язык                  | Python 3.10             |
| API                   | Flask                   |
| Контейнеризация       | Docker / Docker Compose |
| Local LLM             | Ollama                  |
| Основная модель       | Qwen3:4b                |
| Дополнительные модели | Gemma3:4b, Llama3.2:3b  |
| RAG                   | Sentence Transformers   |
| Embeddings            | all-MiniLM-L6-v2        |
| Memory                | SQLite                  |
| Observability         | Langfuse                |
| API testing           | Postman                 |

---

## 4. Структура проекта

```text
multy_agent_system/
│
├── agents/
│   ├── tutor/
│   │   ├── SOUL.md
│   │   ├── BEHAVIOR.md
│   │   ├── RULES.md
│   │   └── skills/
│   │       └── explain_concept.md
│   │
│   └── assessment/
│       ├── SOUL.md
│       ├── BEHAVIOR.md
│       ├── RULES.md
│       └── skills/
│           ├── generate_task.md
│           ├── evaluate_answer.md
│           └── give_feedback.md
│
├── orchestrator/
│   ├── app.py
│   ├── router.py
│   └── langfuse_config.py
│
├── tutor/
│   ├── app.py
│   └── agent.py
│
├── assessment/
│   ├── app.py
│   └── agent.py
│
├── tools/
│   ├── registry.py
│   ├── selector.py
│   ├── knowledge.py
│   └── math_tools.py
│
├── rag/
│   └── rag.py
│
├── rag_data/
│   ├── math.md
│   └── algebra.md
│
├── memory/
│   └── store.py
│
├── shared/
│   ├── ollama.py
│   ├── skill_loader.py
│   └── tokens.py
│
├── evals/
│   ├── run_eval.py
│   └── eval_results.json
│
├── Dockerfile
├── docker-compose.yml
├── EVALUATION.md
└── ARCHITECTURE.md
```

---

## 5. Запуск

Для запуска проекта необходимо установить Docker Desktop.

После получения проекта:

```powershell
docker compose up --build
```

Основные сервисы:

```text
Orchestrator  http://localhost:5000
Tutor         http://localhost:5001
Assessment    http://localhost:5002
Ollama        http://localhost:11434
```

---

## 6. Tutor API

Endpoint:

```text
POST http://localhost:5001/run
```

Пример:

```json
{
    "input": "Объясни, что такое производная",
    "student_id": "ivan",
    "session_id": "session-1"
}
```

Tutor:

1. загружает system identity и skill;
2. ищет релевантный контекст в RAG;
3. загружает memory;
4. формирует prompt;
5. отправляет запрос в Ollama;
6. сохраняет сообщение в memory;
7. возвращает ответ студенту.

---

## 7. Assessment API

Endpoint:

```text
POST http://localhost:5002/run
```

Пример генерации задания:

```json
{
    "input": "Создай задачу по квадратным уравнениям",
    "skill": "generate_task",
    "student_id": "ivan",
    "session_id": "session-2"
}
```

Пример проверки ответа:

```json
{
    "input": "Проверь мой ответ. Уравнение x² - 10x + 16 = 0. Я получил x = 3 и x = 5.",
    "skill": "evaluate_answer",
    "student_id": "ivan",
    "session_id": "session-3"
}
```

---

## 8. Orchestrator API

Основная точка входа пользователя:

```text
POST http://localhost:5000/handle
```

Пример:

```json
{
    "input": "Объясни, что такое производная",
    "student_id": "ivan",
    "session_id": "session-4"
}
```

Orchestrator:

1. получает пользовательский запрос;
2. передаёт его Router;
3. Router выбирает Tutor или Assessment;
4. запрос отправляется выбранному агенту;
5. ответ возвращается пользователю;
6. информация о выполнении записывается в Langfuse.

---

## 9. Router

Router использует детерминированные правила.

Примеры:

```text
"Объясни производную"
        ↓
      Tutor

"Что такое интеграл?"
        ↓
      Tutor

"Какие ошибки я недавно допускал?"
        ↓
      Tutor

"Создай задачу по квадратным уравнениям"
        ↓
    Assessment

"Проверь мой ответ"
        ↓
    Assessment
```

Такой подход выбран из-за небольшого количества агентов и сценариев.

---

## 10. Tools

В системе используется Tool Registry.

Зарегистрированы:

```text
search_knowledge
solve_quadratic
```

### search_knowledge

Использует образовательную базу знаний:

```text
rag_data/math.md
rag_data/algebra.md
```

### solve_quadratic

Решает:

```text
ax² + bx + c = 0
```

и возвращает:

* discriminant;
* roots;
* root count.

При оценивании математический результат инструмента используется как authoritative result.

---

## 11. Memory

Память реализована через SQLite.

База хранится внутри Docker volume:

```text
memory_data
```

Таким образом Tutor и Assessment используют одну и ту же базу.

### Short-term memory

Хранит:

* сообщения;
* session_id;
* student_id;
* роль;
* агента;
* время.

### Long-term memory

Хранит:

* assessment history;
* learning progress;
* mistakes;
* student facts.

Пример:

```text
student_id: ivan
topic: quadratic_equation
mastery_score: 0.0%
```

---

## 12. Shared Memory

Память доступна обоим агентам.

Пример сценария:

```text
Student
   |
   v
Assessment
   |
   v
проверка ответа
   |
   v
Shared SQLite
   |
   v
сохранение ошибки
   |
   v
Tutor
   |
   v
новый запрос
"Какие ошибки я недавно допускал?"
```

В результате Tutor может использовать информацию, сохранённую Assessment в предыдущей сессии.

---

## 13. RAG

RAG используется для поиска учебной информации.

Текущая база:

```text
rag_data/
├── math.md
└── algebra.md
```

Для embeddings используется:

```text
all-MiniLM-L6-v2
```

Основной процесс:

```text
User request
      ↓
Embedding
      ↓
Similarity search
      ↓
Relevant document
      ↓
Tutor prompt
      ↓
Ollama
```

---

## 14. Agent Skills

Каждый агент имеет декларативные Markdown-файлы.

Например:

```text
agents/tutor/skills/explain_concept.md
```

Skill определяет:

* цель;
* последовательность действий;
* структуру результата.

Assessment поддерживает:

```text
generate_task
evaluate_answer
give_feedback
```

Это позволяет изменять поведение агента без переписывания основной логики Python.

---

## 15. Local LLM

Основная модель системы:

```text
Qwen3:4b
```

LLM запускается локально через Ollama.

Дополнительно были протестированы:

```text
Gemma3:4b
Llama3.2:3b
```

Сравнение моделей приведено в:

```text
EVALUATION.md
```

---

## 16. Benchmark

Для трёх моделей использовались одинаковые тесты:

```text
1. explain_derivative
2. generate_quadratic_task
3. evaluate_quadratic_answer
```

Измерялась latency.

Полученные результаты:

| Модель      | Средняя latency |
| ----------- | --------------: |
| Qwen3:4b    |        156.25 s |
| Gemma3:4b   |          8.58 s |
| Llama3.2:3b |          6.79 s |

Помимо времени, анализировалась корректность ответов.

Подробные результаты находятся в:

```text
eval_results.json
```

---

## 17. Langfuse

Langfuse используется для observability.

Orchestrator передаёт:

```text
input
output
model
token usage
agent
student_id
session_id
```

Это позволяет отслеживать:

* запросы;
* выбранного агента;
* latency;
* token usage;
* результаты выполнения.

---

## 18. Isolation

Основные компоненты запускаются в отдельных Docker-контейнерах:

```text
orchestrator
tutor
assessment
ollama
```

Связь между сервисами выполняется через Docker network.

Пример:

```text
orchestrator → tutor:5001
orchestrator → assessment:5002

tutor → ollama:11434
assessment → ollama:11434
```

---

## 19. Framework consideration

Для реализации workflow рассматривались специализированные frameworks, включая LangGraph.

Для текущей версии выбран собственный лёгкий Orchestrator, поскольку система содержит небольшое количество агентов и простую маршрутизацию.

При увеличении количества агентов и состояний Router может быть заменён на framework-based workflow.

---

## 20. Проверка системы

### Проверка контейнеров

```powershell
docker compose ps
```

Все сервисы должны находиться в состоянии:

```text
running
```

### Проверка Router

Пример:

```powershell
docker exec -w /app/orchestrator orchestrator python -c "from router import route; print(route('Объясни производную'))"
```

Ожидаемый результат:

```text
tutor
```

Проверка Assessment:

```powershell
docker exec -w /app/orchestrator orchestrator python -c "from router import route; print(route('Создай задачу по квадратным уравнениям'))"
```

Ожидаемый результат:

```text
assessment
```

### Проверка памяти

```powershell
docker exec assessment python -c "from memory.store import MemoryStore; m=MemoryStore(); print(m.get_student_context('ivan','session-4'))"
```

Проверка должна показывать сохранённые assessment history, progress и mistakes после соответствующих запросов.

---

## 21. Документация

Дополнительные документы проекта:

```text
ARCHITECTURE.md
```

Описание архитектуры, агентов, RAG, tools, memory и Docker isolation.

```text
EVALUATION.md
```

Описание benchmark, моделей, тестов и результатов.

---

## 22. Итог

Проект реализует локальную multi-agent систему со следующими основными компонентами:

```text
Orchestrator
Router
Tutor Agent
Assessment Agent
Skills
Tools
RAG
Shared Memory
Ollama
Langfuse
Docker
```

Система позволяет:

* распределять запросы между специализированными агентами;
* использовать локальную LLM;
* применять RAG;
* использовать инструменты для математических вычислений;
* сохранять историю обучения;
* передавать информацию о студенте между агентами;
* отслеживать работу системы через Langfuse;
* сравнивать несколько локальных моделей.
