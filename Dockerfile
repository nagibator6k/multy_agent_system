FROM python:3.10

WORKDIR /app


RUN pip install flask requests sentence-transformers numpy langfuse tiktoken

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "orchestrator/app.py"]