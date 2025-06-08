FROM python:3.11-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir wandb

ENV REPO_URL=https://github.com/IlliaJa/Warchest.git
ENV SCRIPT_PATH=reinforce.py

CMD bash -c "\
  rm -rf /app/code && \
  git clone $REPO_URL /app/code && \
  cd /app/code && \
  pip install --no-cache-dir -r requirements.txt && \
  python $SCRIPT_PATH"
