FROM nvidia/cuda:11.0.3-base-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

### Install openjdk (java)
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn
ENV TZ=Asia/Seoul
RUN set -e; \
    apt-get update; \
	apt-get install -y --no-install-recommends software-properties-common; \
	apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 0xB1998361219BD9C9; \
	apt-add-repository 'deb http://repos.azulsystems.com/debian stable main'; \
	apt-get update; \
	apt-get install -y --no-install-recommends openjdk-11-jdk; \
	apt-get install -y --no-install-recommends python3.7-dev python3-pip cmake \
	gcc \
	build-essential \
    curl \
    ghostscript \
    git \
    libffi-dev \
    libfreetype6-dev \
    libfribidi-dev \
    libharfbuzz-dev \
    libjpeg-turbo-progs \
    libjpeg8-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    libssl-dev \
    libsqlite3-dev \
    libtiff5-dev \
    libwebp-dev \
    netpbm \
    ninja-build \
    sudo \
    tcl8.6-dev \
    tk8.6-dev \
    wget \
    xvfb \
    zlib1g-dev; \
	# Cleanup cache
	apt-get clean; \
	rm -rf /var/tmp/* /tmp/* /var/lib/apt/lists/*


# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1


### Initialize project 
WORKDIR /app
RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install setuptools wheel cython
RUN python3.7 -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
COPY requirements.txt .
RUN python3.7 -m pip install -r requirements.txt
RUN python3.7 -m pip install --upgrade "protobuf<=3.20.1"
RUN echo 1
COPY . /app

# Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
RUN useradd appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["python3.7", "-u", "manage.py", "runserver", "0.0.0.0:8000"]
