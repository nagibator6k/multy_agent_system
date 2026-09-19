import sqlite3
from pathlib import Path
from typing import Any


class MemoryStore:
    """
    SQLite-based memory for the multi-agent system.

    Short-term memory:
        recent conversation messages.

    Long-term memory:
        student progress,
        assessment history,
        mistakes,
        persistent facts.
    """

    def __init__(
        self,
        db_path: str = "/app/data/memory.db",
    ):
        self.db_path = db_path

        Path(self.db_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._init_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=30,
        )

        connection.row_factory = sqlite3.Row

        connection.execute(
            "PRAGMA busy_timeout = 30000"
        )

        connection.execute(
            "PRAGMA journal_mode = WAL"
        )

        return connection

    def _init_database(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    agent TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_messages_student
                ON messages(student_id, created_at);

                CREATE TABLE IF NOT EXISTS assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    skill TEXT NOT NULL,
                    request TEXT NOT NULL,
                    response TEXT NOT NULL,
                    score REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_assessments_student
                ON assessments(student_id, created_at);

                CREATE TABLE IF NOT EXISTS mistakes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_mistakes_student
                ON mistakes(student_id, created_at);

                CREATE TABLE IF NOT EXISTS progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    mastery_score REAL NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(student_id, topic)
                );

                CREATE TABLE IF NOT EXISTS student_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(student_id, fact_key)
                );
                """
            )

    def save_message(
        self,
        student_id: str,
        session_id: str,
        role: str,
        content: str,
        agent: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages (
                    student_id,
                    session_id,
                    role,
                    content,
                    agent
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    student_id,
                    session_id,
                    role,
                    content,
                    agent,
                ),
            )

    def get_recent_messages(
        self,
        student_id: str,
        session_id: str,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    role,
                    content,
                    agent,
                    created_at
                FROM messages
                WHERE student_id = ?
                  AND session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (
                    student_id,
                    session_id,
                    limit,
                ),
            ).fetchall()

        messages = [dict(row) for row in rows]

        messages.reverse()

        return messages

    def save_assessment(
        self,
        student_id: str,
        session_id: str,
        skill: str,
        request: str,
        response: str,
        score: float | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO assessments (
                    student_id,
                    session_id,
                    skill,
                    request,
                    response,
                    score
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    student_id,
                    session_id,
                    skill,
                    request,
                    response,
                    score,
                ),
            )

    def save_mistake(
        self,
        student_id: str,
        topic: str,
        description: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO mistakes (
                    student_id,
                    topic,
                    description
                )
                VALUES (?, ?, ?)
                """,
                (
                    student_id,
                    topic,
                    description,
                ),
            )

    def save_progress(
        self,
        student_id: str,
        topic: str,
        mastery_score: float,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO progress (
                    student_id,
                    topic,
                    mastery_score
                )
                VALUES (?, ?, ?)

                ON CONFLICT(student_id, topic)
                DO UPDATE SET
                    mastery_score = excluded.mastery_score,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    student_id,
                    topic,
                    mastery_score,
                ),
            )

    def save_fact(
        self,
        student_id: str,
        key: str,
        value: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO student_facts (
                    student_id,
                    fact_key,
                    fact_value
                )
                VALUES (?, ?, ?)

                ON CONFLICT(student_id, fact_key)
                DO UPDATE SET
                    fact_value = excluded.fact_value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    student_id,
                    key,
                    value,
                ),
            )

    def get_student_context(
        self,
        student_id: str,
        session_id: str,
    ) -> str:

        recent_messages = self.get_recent_messages(
            student_id=student_id,
            session_id=session_id,
            limit=6,
        )

        with self._connect() as connection:
            progress_rows = connection.execute(
                """
                SELECT
                    topic,
                    mastery_score,
                    updated_at
                FROM progress
                WHERE student_id = ?
                ORDER BY updated_at DESC
                LIMIT 10
                """,
                (student_id,),
            ).fetchall()

            mistake_rows = connection.execute(
                """
                SELECT
                    topic,
                    description,
                    created_at
                FROM mistakes
                WHERE student_id = ?
                ORDER BY id DESC
                LIMIT 10
                """,
                (student_id,),
            ).fetchall()

            assessment_rows = connection.execute(
                """
                SELECT
                    skill,
                    score,
                    created_at
                FROM assessments
                WHERE student_id = ?
                ORDER BY id DESC
                LIMIT 10
                """,
                (student_id,),
            ).fetchall()

            fact_rows = connection.execute(
                """
                SELECT
                    fact_key,
                    fact_value
                FROM student_facts
                WHERE student_id = ?
                ORDER BY updated_at DESC
                LIMIT 10
                """,
                (student_id,),
            ).fetchall()

        sections: list[str] = []

        if recent_messages:
            lines = []

            for message in recent_messages:
                lines.append(
                    f"{message['role']}: "
                    f"{message['content']}"
                )

            sections.append(
                "RECENT CONVERSATION:\n"
                + "\n".join(lines)
            )

        if progress_rows:
            lines = []

            for row in progress_rows:
                lines.append(
                    f"- {row['topic']}: "
                    f"{row['mastery_score']:.1f}%"
                )

            sections.append(
                "LEARNING PROGRESS:\n"
                + "\n".join(lines)
            )

        if mistake_rows:
            lines = []

            for row in mistake_rows:
                lines.append(
                    f"- {row['topic']}: "
                    f"{row['description']}"
                )

            sections.append(
                "KNOWN MISTAKES:\n"
                + "\n".join(lines)
            )

        if assessment_rows:
            lines = []

            for row in assessment_rows:
                score = row["score"]

                score_text = (
                    f"{score:.1f}%"
                    if score is not None
                    else "not recorded"
                )

                lines.append(
                    f"- {row['skill']}: {score_text}"
                )

            sections.append(
                "ASSESSMENT HISTORY:\n"
                + "\n".join(lines)
            )

        if fact_rows:
            lines = []

            for row in fact_rows:
                lines.append(
                    f"- {row['fact_key']}: "
                    f"{row['fact_value']}"
                )

            sections.append(
                "STUDENT FACTS:\n"
                + "\n".join(lines)
            )

        if not sections:
            return (
                "No stored memory is available "
                "for this student."
            )

        return "\n\n".join(sections)