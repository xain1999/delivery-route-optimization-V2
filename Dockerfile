# <<dockerfile>>
FROM python:3.12-slim

# run as an unprivileged user
RUN useradd --create-home routing
WORKDIR /app

COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY vrpbook/ vrpbook/
COPY routing_service/ routing_service/

USER routing
ENV ROUTING_MAX_WORKERS=2 \
    ROUTING_MAX_TIME_LIMIT_S=120 \
    ROUTING_SYNC_LIMIT_S=5 \
    ROUTING_MAX_PENDING=8
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c \
  "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "routing_service.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--log-level", "info"]
# <<end_dockerfile>>
