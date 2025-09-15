FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl nano build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir jupyterlab ipython

# (이미 있다면) requirements.txt 설치 단계는 기존대로 유지
# COPY requirements.txt /tmp/requirements.txt
# RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

WORKDIR /workspace