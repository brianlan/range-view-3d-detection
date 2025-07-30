# What Matters in Range View 3D Object Detection [CoRL 2024]

[Paper](https://openreview.net/forum?id=EifoVoIyd5)

## Datasets

- Argoverse 2.
- Waymo Open.

You will need to download both of these datasets in their entirety. After downloading, run the converters in `./converters`. This will output each dataset in the format that our codebase expects.

## Environment

You will need to install conda and install our conda environment. Please follow [INSTALL.md](INSTALL.md):

This will install the environment to run the codebase. Note: This environment has only been tested on Ubuntu 20.04 using A40 gpus.

Additionally, to use weighted NMS you will need to install: https://github.com/Abyssaledge/TorchEx.

## Training Script

The entrypoint is found in `scripts`. For example, to train our SOTA comparison model for Argoverse 2, you would run:

```
PYTHONPATH=$(pwd) WANDB_MODE=disabled python scripts/train.py \
    experiment=rv-av2 \
    ++trainer.max_epochs=20 \
    ++dataset.root_dir=/ssd3/datasets/argoverse2/mini-processed/sensor \
    ++trainer.devices=1 \
    ++trainer.logger._target_=pytorch_lightning.loggers.TensorBoardLogger \
    '++hydra.run.dir=experiments/${now:%Y-%m-%d-%H-%M-%S}'
```

or 

`bash train.sh rv-av2 4 20 1`.
