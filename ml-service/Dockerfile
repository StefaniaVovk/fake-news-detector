FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app
COPY start.sh .

RUN mkdir -p /app/output/logreg /app/output/bert-tiny && \
    chmod -R 777 /app/output

RUN chmod +x start.sh

EXPOSE 8000
CMD ["./start.sh"]
