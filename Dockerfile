FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    python3 \
    python3-pip \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Create a working directory
RUN mkdir /app
WORKDIR /app

# cope requirements
COPY ./requirements.txt /app/requirements.txt

# install requirements
RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt

# copy rest
COPY . /app/.

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
