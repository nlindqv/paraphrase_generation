FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime
WORKDIR /workspace
#COPY . .
ADD requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
