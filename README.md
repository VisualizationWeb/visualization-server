<p align="center">
  <img width="256" height="256" src="https://github.com/VisualizationWeb/visualization-resources/blob/main/icon.png?raw=true">
  <h1 align="center">visualization-server</h1>
</p>

<br>

# 실행 방법

> 이 프로젝트를 실행하기 위해선 docker가 설치되어 있어야 합니다.
> **[docker](https://www.docker.com/get-started)** 사이트를 방문하여 OS에 맞는 docker 런타임을 설치해주세요.

> Ubuntu 터미널에서 설치를 진행한다면, `sudo apt install docker` 명령어로 설치를 진행해주세요.

<br>

### 1. 레포지토리 다운로드

```bash
$ git clone https://github.com/VisualizationWeb/visualization-server.git
```

`git` 명령어를 통해 레포지토리를 복제(clone)합니다.

<br>

### 2. Docker 컨테이너 실행

```bash
$ cd visualization-server
$ docker-compose up -d
```

다운받은 `visualization-server` 폴더로 이동한 뒤, `docker-compose` 명령어로 컨테이너를 생성 및 실행시킵니다.

<br>

### 3. 실행 확인

```bash
$ curl -s -l localhost:8000
```

`visualization-server` 는 `8000`번 포트를 기본값으로 사용합니다. 브라우저 또는 커맨드라인을 통해 접속이 원활한지 확인해주세요.
