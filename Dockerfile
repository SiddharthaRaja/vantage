# The benchmark client, not the model server. Deliberately small: it must be
# runnable next to a GPU box without dragging CUDA along.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY configs ./configs
COPY data ./data
COPY scripts ./scripts

# Results are written here; mount a volume to keep them.
VOLUME ["/app/results"]

ENTRYPOINT ["vantage"]
CMD ["demo"]
