FROM tensorflow/tensorflow:latest-gpu

ARG VOLUME_DIR

RUN mkdir ${VOLUME_DIR}

COPY . ${VOLUME_DIR}

WORKDIR ${VOLUME_DIR}

RUN /usr/bin/python3 -m pip install --upgrade pip

RUN pip3 install -r requirements.txt
