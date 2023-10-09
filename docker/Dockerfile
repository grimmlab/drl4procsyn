FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt-get update
RUN apt-get -y install tmux

COPY ./requirements.txt .
RUN pip install -r ./requirements.txt