# Architecture

## 1. Обзор системы

Проект представляет собой локальную multi-agent систему для обучения студентов.

Система состоит из двух специализированных агентов:

* **Tutor Agent** — объясняет учебные понятия и использует базу знаний;
* **Assessment Agent** — создаёт учебные задания, проверяет ответы и предоставляет обратную связь.

Для управления взаимодействием используется отдельный **Orchestrator**, который определяет, какой агент должен обработать запрос.

Все LLM работают локально через **Ollama**.

---

## 2. Общая архитектура

```text
                         Student
                            |
                            v
                    +---------------+
                    |  Orchestrator |
                    |    :5000      |
                    +-------+-------+
                            |
                            v
                       +---------+
                       |  Router |
                       +----+----+
                            |
                 +----------+----------+
                 |                     |
                 v                     v
          +-------------+       +---------------+
          | Tutor Agent |       | Assessment    |
          |    :5001    |       | Agent :5002   |
          +------+------+       +-------+-------+
                 |                      |
                 v                      v
             +-------+            +----------+
             |  RAG  |            |   Tools  |
             +-------+            +----------+
                 |                      |
                 +----------+-----------+
                            |
                            v
                    +---------------+
                    | Shared Memory |
                    |    SQLite     |
                    +-------+-------+
                            |
                            v
                       +---------+
                       | Ollama  |
                       | Qwen3   |
                       +---------+
                            |
                            v
                       +---------+
                       | Langfuse|
                       +---------+
```

---

## 3. Orchestrator

Orchestrator является центральной точкой входа в систему.

Endpoint:

```text
POST /handle
```

Порт:

```text
5000
```

Orchestrator выполняет следующие действия:

1. получает запрос пользователя;
2. получает `student_id` и `session_id`;
3. передаёт запрос Router;
4. Router выбирает агента;
5. Orchestrator отправляет запрос выбранному агенту;
6. получает результат;
7. возвращает ответ пользователю;
8. записывает информацию о запросе в Langfuse.

Orchestrator не выполняет учебную задачу самостоятельно.

---

## 4. Router

Router отвечает за определение типа пользовательского запроса.

В текущей минимальной реализации используется детерминированный rule-based подход.

Примеры:

```text
"Объясни производную"
        ↓
      Tutor

"Что такое интеграл?"
        ↓
      Tutor

"Создай задачу по квадратным уравнениям"
        ↓
    Assessment

"Проверь мой ответ"
        ↓
    Assessment
```

Rule-based Router выбран из-за небольшого количества агентов и сценариев.

Для текущего прототипа дополнительная LLM для маршрутизации не требуется.

---

## 5. Tutor Agent

Tutor Agent специализируется на объяснении учебного материала.

Порт:

```text
5001
```

Основные обязанности:

* объяснение понятий;
* адаптация объяснения под студента;
* использование RAG;
* использование student memory;
* формирование примеров;
* проверка понимания.

Tutor использует следующие компоненты:

```text
SOUL.md
BEHAVIOR.md
RULES.md
explain_concept.md
search_knowledge
MemoryStore
```

Tutor не занимается оцениванием ответов как основной функцией.

---

## 6. Assessment Agent

Assessment Agent отвечает за оценивание.

Порт:

```text
5002
```

Поддерживаемые skills:

```text
generate_task
evaluate_answer
give_feedback
```

Основные функции:

* генерация учебных заданий;
* проверка ответов;
* анализ ошибок;
* выставление оценки;
* сохранение результатов обучения.

Assessment использует:

```text
SOUL.md
BEHAVIOR.md
RULES.md
generate_task.md
evaluate_answer.md
give_feedback.md
ToolSelector
ToolRegistry
MemoryStore
```

---

## 7. Skills

Skills представляют собой декларативные `.md` файлы с описанием поведения агента.

Пример структуры:

```text
agents/
├── tutor/
│   ├── SOUL.md
│   ├── BEHAVIOR.md
│   ├── RULES.md
│   └── skills/
│       └── explain_concept.md
│
└── assessment/
    ├── SOUL.md
    ├── BEHAVIOR.md
    ├── RULES.md
    └── skills/
        ├── generate_task.md
        ├── evaluate_answer.md
        └── give_feedback.md
```

Разделение на отдельные файлы позволяет изменять поведение агента без изменения Python-кода.

---

## 8. Tools

Для работы с внешними функциями используется `ToolRegistry`.

В текущей реализации зарегистрированы:

```text
search_knowledge
solve_quadratic
```

### search_knowledge

Используется для поиска информации в образовательной базе знаний.

### solve_quadratic

Решает квадратное уравнение:

```text
ax² + bx + c = 0
```

Инструмент возвращает:

* дискриминант;
* корни;
* количество корней.

Математический результат инструмента рассматривается как источник истины при проверке ответа студента.

---

## 9. RAG

RAG используется для предоставления Tutor Agent дополнительного учебного контекста.

Текущая база знаний содержит Markdown-файлы:

```text
rag_data/
├── math.md
└── algebra.md
```

Поиск выполняется с использованием:

```text
sentence-transformers
all-MiniLM-L6-v2
```

Общий процесс:

```text
Student request
      ↓
Embedding query
      ↓
Similarity search
      ↓
Relevant document
      ↓
Tutor prompt
      ↓
LLM response
```

RAG позволяет агенту использовать заранее подготовленные учебные материалы вместо генерации ответа только на основе параметров LLM.

---

## 10. Memory

В системе реализована shared memory на базе SQLite.

Путь базы данных внутри контейнеров:

```text
/app/data/memory.db
```

Для Tutor и Assessment используется общий Docker volume:

```text
memory_data
```

Таким образом, агенты имеют доступ к общей памяти студента.

---

## 11. Short-term memory

Short-term memory хранит сообщения текущей учебной сессии.

Для каждой записи используются:

```text
student_id
session_id
role
content
agent
created_at
```

Пример:

```text
student_id = ivan
session_id = session-5
role = user
content = "Какие ошибки я недавно допускал?"
agent = tutor
```

Tutor получает несколько последних сообщений текущей сессии при формировании prompt.

---

## 12. Long-term memory

Long-term memory предназначена для сохранения результатов обучения между сессиями.

Хранятся:

```text
assessment history
learning progress
mistakes
student facts
```

Например:

```text
student_id: ivan
topic: quadratic_equation
mastery_score: 0.0
```

или:

```text
topic: quadratic_equation
description:
Assessment score: 0.0%.
```

Это позволяет использовать опыт предыдущих взаимодействий.

---

## 13. Shared memory между агентами

Одна из важных особенностей системы — Tutor и Assessment используют одну базу памяти.

Пример:

```text
Assessment
    |
    | проверяет ответ
    v
Shared SQLite
    |
    | сохраняет ошибку
    v
Tutor
    |
    | новый запрос
    v
"Какие ошибки я допускал?"
```

В тесте студент `ivan` получил оценку `0%` за неправильное решение квадратного уравнения.

После этого Tutor в другой сессии смог получить информацию о предыдущей ошибке и использовать её в ответе.

Это демонстрирует обмен состоянием между специализированными агентами.

---

## 14. Изоляция

Каждый основной компонент системы запускается в Docker-контейнере.

Используются следующие сервисы:

```text
orchestrator
tutor
assessment
ollama
```

Docker Compose отвечает за запуск и сетевое взаимодействие контейнеров.

Пример:

```text
orchestrator → tutor:5001
orchestrator → assessment:5002
tutor       → ollama:11434
assessment  → ollama:11434
```

Такое разделение позволяет изолировать компоненты системы и независимо изменять агентов.

---

## 15. Local LLM

LLM запускается локально через Ollama.

Использовавшаяся основная модель:

```text
Qwen3:4b
```

Дополнительно были протестированы:

```text
Gemma3:4b
Llama3.2:3b
```

Все модели работают локально, без необходимости отправлять учебные запросы во внешний LLM API.

---

## 16. Langfuse

Langfuse используется для наблюдаемости системы.

Orchestrator создаёт observation для обработки запроса.

В Langfuse передаются:

```text
input
output
model
token usage
agent
student_id
session_id
```

Это позволяет отслеживать выполнение запросов и анализировать:

* latency;
* token usage;
* выбранного агента;
* входные и выходные данные;
* отдельные вызовы системы.

---

## 17. Почему два агента

В системе использованы два агента с разными обязанностями.

### Tutor

Отвечает за:

```text
объяснение
обучение
RAG
работу с учебным контекстом
```

### Assessment

Отвечает за:

```text
генерацию заданий
проверку ответов
оценивание
анализ ошибок
feedback
```

Разделение позволяет уменьшить количество обязанностей одного агента и задавать каждому агенту специализированные инструкции и skills.

---

## 18. Рассмотрение framework

Для построения multi-agent workflow рассматривалось использование специализированных frameworks, например LangGraph.

LangGraph предоставляет средства для построения stateful workflows и переходов между состояниями.

Для текущего проекта выбран собственный лёгкий Orchestrator, поскольку система содержит только двух агентов и небольшое количество сценариев.

Такой вариант уменьшает количество зависимостей и позволяет явно контролировать маршрутизацию запросов.

При увеличении количества агентов собственный Router может быть заменён на framework-based workflow.

---

## 19. Структура проекта

Основные директории:

```text
multy_agent_system/
│
├── agents/
│   ├── tutor/
│   └── assessment/
│
├── assessment/
├── tutor/
├── orchestrator/
│
├── tools/
├── rag/
├── rag_data/
├── memory/
├── shared/
├── evals/
│
├── Dockerfile
├── docker-compose.yml
├── EVALUATION.md
└── ARCHITECTURE.md
```

---

## 20. Основной поток запроса

Полный жизненный цикл запроса выглядит следующим образом:

```text
1. Student sends request
        ↓
2. Orchestrator receives request
        ↓
3. Router determines agent
        ↓
4. Selected agent loads:
   - system identity
   - behavior
   - rules
   - skill
        ↓
5. Agent retrieves:
   - RAG context
   - tools
   - student memory
        ↓
6. Prompt is created
        ↓
7. Local LLM generates response
        ↓
8. Response is cleaned if necessary
        ↓
9. Memory is updated
        ↓
10. Langfuse receives observability data
        ↓
11. Response is returned to Student
```

---

## 21. Основные технологии

| Компонент         | Технология              |
| ----------------- | ----------------------- |
| Language          | Python 3.10             |
| API               | Flask                   |
| Containers        | Docker / Docker Compose |
| LLM runtime       | Ollama                  |
| Main LLM          | Qwen3:4b                |
| Additional models | Gemma3:4b, Llama3.2:3b  |
| Embeddings        | Sentence Transformers   |
| RAG               | all-MiniLM-L6-v2        |
| Memory            | SQLite                  |
| LLM observability | Langfuse                |
| API testing       | Postman                 |

---

## 22. Итог

Архитектура проекта представляет собой небольшую специализированную multi-agent систему.

Основные компоненты имеют отдельные роли:

```text
Orchestrator → координация
Router       → маршрутизация
Tutor        → обучение
Assessment   → оценивание
Tools        → вычисления и поиск
RAG          → учебный контекст
Memory       → состояние студента
Ollama       → локальная LLM
Langfuse     → observability
```

Такая архитектура достаточно проста для локального запуска и одновременно демонстрирует основные элементы LLM-based multi-agent системы.
