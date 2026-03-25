FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    EAGLE_DOCKER=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    binutils \
    patchelf \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libjpeg62-turbo \
    libsm6 \
    libxext6 \
    libxrender1 \
    patchelf \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN grep -Ev '^(altgraph|appnope|macholib|pyinstaller|pyinstaller-hooks-contrib|torch|torchvision|torchaudio)==.*' /tmp/requirements.txt > /tmp/requirements-docker.txt \
 && python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install -r /tmp/requirements-docker.txt \
 && python -m pip install "numpy<2" \
 && python -m pip install \
      --extra-index-url https://download.pytorch.org/whl/cu113 \
      torch==1.12.1+cu113 \
      torchvision==0.13.1+cu113 \
 && find /usr/local/lib/python3.10/site-packages/torch -name "*.so*" -exec patchelf --clear-execstack {} \; \
 && readelf -W -l /usr/local/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so | grep GNU_STACK \
 && ! readelf -W -l /usr/local/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so | grep -q "RWE"

COPY . /app

RUN chmod +x /app/ffmpeg/AMD/ffmpeg

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--global.developmentMode=false", "--server.fileWatcherType=none"]
