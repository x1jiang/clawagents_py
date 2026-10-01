FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && useradd --uid 10001 --no-create-home appuser
COPY examples ./examples
USER 10001
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn examples.hip_dislocation.web.app:app --host 0.0.0.0 --port ${PORT:-8080} --no-access-log"]
