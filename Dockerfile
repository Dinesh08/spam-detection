FROM ubuntu:18.04
RUN apt-get -y update
RUN apt-get install -y python3.6 python3-pip python3-dev build-essential gcc \libsnmp-dev snmp-mibs-downloader
RUN pip3 install --upgrade pip
RUN apt-get -y install libc-dev
RUN apt-get -y install build-essential
RUN pip install --upgrade --no-cache-dir pip setuptools==49.6.0
RUN pip install -U pip
RUN mkdir /app
COPY . /app
WORKDIR /app
ENV user=
RUN pip3 install -r requirements.txt
RUN python3 -m nltk.downloader stopwords
ENTRYPOINT ["python3","final.py"]