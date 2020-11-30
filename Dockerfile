FROM uhub.service.ucloud.cn/uai_dockers/cambricon_pytorch:v1.0
RUN /bin/bash -c " source /opt/cambricon/pytorch/src/catch/venv/pytorch/bin/activate && pip install scikit-learn pretrainedmodels imgaug tqdm pillow easydict prefetch_generator flask &&  pip install --upgrade scipy"
COPY . /root
CMD  . /opt/cambricon/pytorch/src/catch/venv/pytorch/bin/activate && exec python -u /root/server.py
