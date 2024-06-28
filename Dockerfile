FROM ghcr.io/tna76874/binaryaiapibase:latest

COPY main.py /app/main.py
COPY detect.py /app/detect.py
RUN chmod -R +x /app

ENV CNN_API_KEY=test
ENV CNN_MODEL_PATH=data/model.keras

EXPOSE 5000

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["sh", "-c", "/app/entrypoint.sh"]
