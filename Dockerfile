FROM python:3.10

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git libgl1 libmagic1 poppler-utils

COPY requirements.txt /app/requirements.txt
RUN pip install pipreqs && pip install -r requirements.txt

COPY main.py /app/main.py
COPY detect.py /app/detect.py
RUN chmod -R +x /app

ENV CNN_API_KEY=test
ENV CNN_MODEL_PATH=data/model.keras

EXPOSE 5000

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["sh", "-c", "/app/entrypoint.sh"]
