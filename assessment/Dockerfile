FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install flask requests sentence-transformers numpy langfuse tiktoken

CMD ["python", "app.py"]