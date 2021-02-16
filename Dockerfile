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


# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1


### Initialize project
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . /app

# Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
RUN useradd appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
