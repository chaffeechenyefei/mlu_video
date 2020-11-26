FROM uhub.service.ucloud.cn/uai_dockers/cambricon_pytorch:v1.0
COPY . /root
RUN pip install scikit-learn pretrainedmodels imgaug tqdm pillow easydict prefetch_generator flask && \
    pip install --upgrade scipy
ENTRYPOINT ["python", "server.py"]
