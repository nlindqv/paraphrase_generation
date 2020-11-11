FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
WORKDIR /workspace
ADD requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
