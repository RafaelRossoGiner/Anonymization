FROM python:3.7.8

ENV INSTALL_PATH /anonymization_docker
RUN mkdir -p $INSTALL_PATH
WORKDIR $INSTALL_PATH

ADD Pipfile .
RUN mkdir -p .venv

ADD config ${INSTALL_PATH}/config
COPY anonymization_docker ${INSTALL_PATH}/anonymization_docker

RUN apt-get update && apt-get install -y gcc wget
RUN wget https://dlm.mariadb.com/2678574/Connectors/c/connector-c-3.3.3/mariadb-connector-c-3.3.3-debian-bullseye-amd64.tar.gz -O - | tar -zxf - --strip-components=1 -C /usr

RUN pip install --upgrade pip
RUN pip3 install pipenv==2022.1.8

COPY ./requirements.txt $INSTALL_PATH/requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt
RUN rm -rf ~/.cache
RUN rm -rf /tmp/*



CMD gunicorn -b 0.0.0.0:8080 --access-logfile - "anonymization_docker.app:application"