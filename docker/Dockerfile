FROM ubuntu:16.04

RUN apt update && apt upgrade -y
RUN apt install wget build-essential gcc -y

RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.3.21-Linux-x86_64.sh -O miniconda.sh --no-check-certificate
RUN bash miniconda.sh -bf -p /anaconda
RUN rm miniconda.sh
ENV PATH /anaconda/bin:$PATH
RUN pip install --upgrade pip
RUN pip install pytest coverage
RUN conda install numpy pandas scikit-learn joblib -y
