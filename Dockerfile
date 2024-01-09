# Use the specified base image
FROM docker.yard.oa.com:14917/zzihliu/tfccitisbaseimg:cuda12.1-cudnn8-devel-centos7

USER root
# Set the working directory in the container to /vllm
WORKDIR /home/qspace/vllm
ADD ./requirements.txt /home/qspace/vllm/requirements.txt
RUN pip install -r requirements.txt -i https://mirrors.tencent.com/pypi/simple/ --trusted-host mirrors.tencent.com
RUN pip install tfccitispykit==1.3.8 -i https://mirrors.tencent.com/pypi/simple/ --trusted-host mirrors.tencent.com
RUN pip install transformers_stream_generator einops accelerate flash-attn pillow matplotlib aioprometheus -i https://mirrors.tencent.com/pypi/simple/ --trusted-host mirrors.tencent.com


RUN yum install -y centos-release-scl
RUN yum install -y devtoolset-8-gcc-*
RUN echo "source scl_source enable devtoolset-8" >> /etc/bashrc
RUN source /etc/bashrc

COPY . /home/qspace/vllm
