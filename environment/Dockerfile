# hash:sha256:090ce926877f82c796197f2ccc03de0169a8a825e9a7af67ef2989b36425bade
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch-tensorflow-jax:2.1

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    matplotlib==3.8.4 \
    torchaudio==2.3.0 \
    torchvision==0.18.0 \
    wandb==0.17.0
