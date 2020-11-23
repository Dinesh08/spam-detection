FROM ubuntu
RUN apt-get -y update
RUN apt-get install -y python3.6 python3-pip python3-dev python3-venv build-essential gcc \libsnmp-dev snmp-mibs-downloader
RUN apt-get -y install libc-dev
RUN apt-get -y install build-essential
RUN mkdir /app
COPY . /app
WORKDIR /app
ENV user=
RUN pip3 install -r requirements.txt
#RUN python3
#RUN python3 -m nltk.downloader stopwords
CMD ["python3","nltk-stopwords.py"]
ENTRYPOINT ["python3","final.py"]
