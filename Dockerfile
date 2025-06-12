FROM python:3.10-slim
    
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config\
    libmariadb-dev \
    libmariadb-dev-compat \
    libpng-dev \
    zlib1g-dev\
    libwoff-dev\
    libwoff1\
    libfontconfig1\
    libfreetype6\
    fonts-liberation\
    libx11-6\
    libxext6\
    libxrender1\
    && apt-get clean\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip==25.0.1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
   DJANGO_SETTINGS_MODULE=Youtube.settings

EXPOSE 8000

CMD [ "gunicorn","--bind","0.0.0.0:8000","Youtube.wsgi:application" ]




