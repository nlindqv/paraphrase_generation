FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime
WORKDIR ~/paraphrase_generation
COPY . .
RUN pip install -r requirements.txt
