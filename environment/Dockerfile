# hash:sha256:81e823bc11c3f89c3919d86603de33a4d76608f9cc11d76218f3ebb349c97a69
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch-tensorflow-jax:2.1

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    matplotlib==3.8.4 \
    profilehooks \
    pydantic \
    toml \
    torchaudio \
    torchvision \
    wandb==0.17.0
