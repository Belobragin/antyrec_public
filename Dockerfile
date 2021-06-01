FROM python:3.8.3

RUN mkdir /antyrec
WORKDIR /antyrec

COPY alpmapi.py /antyrec/
COPY requirements.txt /antyrec/

COPY utests /antyrec/utests
COPY  alpm /antyrec/alpm

RUN pip install -r requirements.txt
RUN apt update && apt install -y lsof
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev

#EXPOSE 8006

CMD ["python", "alpmapi.py"]





