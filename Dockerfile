FROM jupyter/datascience-notebook

ENV JUPYTER_ENABLE_LAB true
ENV PIP_TARGET /home/jovyan/work/python_packages
ENV PYTHONPATH /home/jovyan/work/python_packages
ENV TORCH_HUB /home/jovyan/work
ENV TORCH_HOME /home/jovyan/work

WORKDIR /home/jovyan/

RUN pip install torch==1.4.0
RUN pip install torchvision==0.5.0
RUN pip install pydicom==1.4.2
RUN pip install pylidc==0.2.1
RUN pip install SimpleITK==1.2.4
RUN pip install torchnet==0.0.4
RUN pip install h5py==2.10.0
RUN pip install tensorboard==1.14.0
RUN pip install ninja==1.9.0.post1
RUN pip install medpy==0.4.0
RUN pip install mdai==0.4.1
RUN pip install timm==0.1.20
RUN pip install pretrainedmodels==0.7.4
RUN pip install efficientnet_pytorch==0.6.3
RUN pip install matplotlib==3.1.2
RUN pip install seaborn==0.9.0
RUN pip install batchgenerators==0.20.1
RUN pip install scikit-image
RUN pip install kornia==0.2.0

RUN pip install git+https://github.com/haven-ai/haven.git
RUN pip install git+https://github.com/lucasb-eyer/pydensecrf.git
RUN pip install pycocotools
RUN pip install nibabel

RUN mkdir -p /home/jovyan/work/python_packages

EXPOSE 8080

COPY --chown=jovyan:users . /home/jovyan/covid

CMD ["/bin/sh", "-c", "start-notebook.sh --ip=0.0.0.0 --port=8080 --no-browser --LabApp.token='' --LabApp.allow_remote_access=True --LabApp.allow_origin='*' --LabApp.disable_check_xsrf=True"]

