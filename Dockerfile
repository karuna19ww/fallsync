FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app
