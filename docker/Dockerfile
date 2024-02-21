# ============================ Multi Region Settings ==========================

ARG REGION

FROM hub.byted.org/base/lab.pytorch:1.10.0.post39 as US-East
# Description: [master] fix torch_iter miss bug
# Publish platform:Git 触发|Branch name:master|Commit ID：67858c6
ENV DATA_CRUISE_CRUISE_VERSION=1.0.0.908
ENV CRUISE_WITH_TRACKING=0

FROM hub.byted.org/base/lab.pytorch:1.10.0.post39 as US-TTP
# Description: [master] [distro] chore: bump version to 0.3.1
# Publish platform:SCM平台|Branch name:master|Commit ID：5161881
ENV DATA_CRUISE_CRUISE_VERSION=2.0.0.125
ENV CRUISE_WITH_TRACKING=0

FROM hub.byted.org/base/lab.pytorch:1.10.0.post39 as US-East-Red
ENV DATA_CRUISE_CRUISE_VERSION=1.0.0.908
ENV CRUISE_WITH_TRACKING=0

FROM hub.byted.org/base/lab.pytorch:1.10.0.post39 as Aliyun_VA
ENV DATA_CRUISE_CRUISE_VERSION=1.0.0.908
ENV CRUISE_WITH_TRACKING=0

FROM hub.byted.org/base/lab.pytorch:1.10.0.post39 as Aliyun_SG
ENV DATA_CRUISE_CRUISE_VERSION=1.0.0.908
ENV CRUISE_WITH_TRACKING=0

FROM hub.byted.org/base/lab.pytorch:1.10.0.post39 as China-North-LF
ENV DATA_CRUISE_CRUISE_VERSION=1.0.0.908
ENV http_proxy="http://sys-proxy-rd-relay.byted.org:8118"
ENV https_proxy="http://sys-proxy-rd-relay.byted.org:8118"
ENV no_proxy="byted.org"
ENV CRUISE_WITH_TRACKING=1

FROM ${REGION}

# ============================ Cruise Dependency ==============================

RUN apt update && apt install python3-opencv=3.2.0+dfsg-6 -y

RUN pip3 install byted-dataloader==0.3.1 \
    jsonargparse[signatures]==4.14 fsspec dill tqdm pyyaml==5.4.1 \
    protobuf==3.20 tensorflow-cpu==2.8 tensorboard==2.8 py-spy tensorflow-io==0.25.0 \
    pyarrow==4.0.1 pandas==1.3.5 timm==0.6.7 transformers==4.21.0 \
    deepspeed==0.7.3 bytedtcc==1.4.2 easydict==1.10 soundfile==0.12.1 --no-cache-dir

# easyguard special pkgs
RUN pip3 install sentencepiece==0.1.97 opencv-python yacs==0.1.8 absl-py bytedabase prettytable huggingface-hub>=0.10.0 decord --no-cache-dir

# anyon
# Description: [master] Merge branch 'feature/enrich_exporter_api' into 'master' add module packing api for exporter See merge request lab/anyon2!105
RUN SCM_NAME=lab.anyon.anyon2 SCM_VERSION=1.0.0.570 SCM_PATH=/opt/tiger/anyon2 && \
    wget -O /tmp/${SCM_NAME}.tar.gz ${LAB_PYTORCH_SCM_SITE}/${SCM_NAME}_${SCM_VERSION}.tar.gz && \
    (mkdir -p ${SCM_PATH} && tar -zxf /tmp/${SCM_NAME}.tar.gz -C ${SCM_PATH} || exit -1) && \
    rm -rf /tmp/${SCM_NAME}.tar.gz

# titan
# RUN pip3 install bytedtitan -i https://bytedpypi.byted.org/simple/ --no-cache-dir
RUN pip3 install http://luban-source.byted.org/repository/scm/data.aml.titan_1.0.0.100.tar.gz --no-cache-dir

# tracking is part of lab.pytorch image

# ============================ Cruise ==============================


ARG CUSTOM_CRUISE_SCM_DOWNLOAD_URL
RUN if [ -n "$CUSTOM_CRUISE_SCM_DOWNLOAD_URL" ] ; then \
        pip3 install --no-cache-dir $CUSTOM_CRUISE_SCM_DOWNLOAD_URL; \
    else \
        pip3 install --no-cache-dir byted-cruise==0.3.4; \
    fi

# pip3 install --no-cache-dir $LAB_PYTORCH_SCM_SITE/data.aml.cruise_$DATA_CRUISE_CRUISE_VERSION.tar.gz; \


ENV http_proxy=
ENV https_proxy=
ENV no_proxy=

ENV USE_NATIVE_HDFS_CLIENT=1

