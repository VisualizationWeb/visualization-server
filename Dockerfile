FROM python:3.8


### Install openjdk (java)
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn
RUN set -e; \
    apt-get update; \
	apt-get install -y --no-install-recommends software-properties-common; \
	apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 0xB1998361219BD9C9; \
	apt-add-repository 'deb http://repos.azulsystems.com/debian stable main'; \
	apt-get update; \
	apt-get install -y --no-install-recommends openjdk-11-jdk; \
	# Cleanup cache
	apt-get clean; \
	rm -rf /var/tmp/* /tmp/* /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1

WORKDIR /code

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /code/

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]