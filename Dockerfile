FROM python:3.11-slim

# Create user 1000 required for HuggingFace Spaces
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install system dependencies (curl, git, bash)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create workspace and reports directories
RUN mkdir -p /home/user/app/workspace /home/user/app/reports /home/user/app/scripts && \
    chown -R user:user /home/user

USER user
COPY --chown=user:user . .

ENV PORT=7860
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
