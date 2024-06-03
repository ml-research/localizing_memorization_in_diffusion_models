FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ARG wandb_key

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# set the working directory and copy everything to the docker file
WORKDIR ./
COPY ./requirements.txt ./

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y git
RUN conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

RUN if [ -z "$wandb_key" ] ; then echo WandB API key not provided ; else wandb login "$wandb_key"; fi
