# Installation Guide

Based on the prefusion environment, we need to do some extra installations.

## Install some infrastructure packages
MAX_JOBS=8 pip install -v --no-build-isolation --use-pep517 \
    hydra-core==1.3.2 \
    hydra-submitit-launcher==1.2.0 \
    pytorch_lightning==2.5.2 \
    tensorflow==2.11.0 \
    wandb==0.13.11

## Install av2 related packages

```
pip install -v --no-build-isolation --use-pep517 av2==0.3.5
```

## Install detectron2

```
MAX_JOBS=16 python -m pip install -v --use-pep517 --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```


## Install the codebase itself, namely torchbox3d

```
pip install -e . # at the root directory of the project
```

## Install TorchEx

```
git clone https://github.com/Abyssaledge/TorchEx.git
cd TorchEx
pip install -e .
```