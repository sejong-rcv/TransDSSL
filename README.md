## [TransDSSL: Transformer based Depth Estimation via Self-Supervised Learning](https://ieeexplore.ieee.org/document/9851497/)
[![IEEE RA-L 2022](https://img.shields.io/badge/-IEEE%20RA--L%202022-blue)](https://ieeexplore.ieee.org/document/9851497) 

## Abstract     
Recently, transformers have been widely adopted for various computer vision tasks and show promising results due to their ability to encode long-range spatial dependencies in an image effectively. However, very few studies on adopting transformers in self-supervised depth estimation have been conducted. When replacing the CNN architecture with the transformer in self-supervised learning of depth, we encounter several problems such as problematic multi-scale photometric loss function when used with transformers and, insufficient ability to capture local details. In this letter, we propose an attention-based decoder module, Pixel-Wise Skip Attention (PWSA), to enhance fine details in feature maps while keeping global context from transformers. In addition, we propose utilizing self-distillation loss with single-scale photometric loss to alleviate the instability of transformer training by using correct training signals. We demonstrate that the proposed model performs accurate predictions on large objects and thin structures that require global context and local details. Our model achieves state-of-the-art performance among the self-supervised monocular depth estimation methods on KITTI and DDAD benchmarks.


## Contents

- [Install](#Install)
    - [Cuda 11](#Cuda-11)
    - [horovod](#horovod)
- [Datasets](#Datasets)
    - [Dense Depth for Autonomous Driving (DDAD)](#Dense-Depth-for-Autonomous-Driving-(DDAD))
    - [KITTI](#KITTI)
- [Swin Transformer(Pretrained-model)](#Swin-Transformer(Pretrained-model))
- [Evaluation](#Evaluation)
- [Models](#Models)
    - [DDAD](#DDAD)
    - [KITTI](#KITTI)
- [Acknowledgement](#Acknowledgement)
- [References](#References)



## Install

You need a machine with recent Nvidia drivers and a GPU with at least 6GB of memory (more for the bigger models at higher resolution). We recommend using docker (see [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker) instructions) to have a reproducible environment. To setup your environment, type in a terminal (only tested in Ubuntu 18.04):

```bash
# if you want to use docker (recommended)
make docker-build
make docker-start-interactive
```
We will list below all commands as if run directly inside our container. To run any of the commands in a container, you can either start the container in interactive mode with `make docker-start-interactive` to land in a shell where you can type those commands, or you can do it in one step:

If you want to use features related to [Weights & Biases (WANDB)](https://www.wandb.com/) (for experiment management/visualization), then you should create associated accounts and configure your shell with the following environment variables:

```bash
export WANDB_ENTITY="something"
export WANDB_API_KEY="something"
```

To enable WANDB logging and AWS checkpoint syncing, you can then set the corresponding configuration parameters in `configs/<your config>.yaml` (cf. [configs/default_config.py](./configs/default_config.py) for defaults and docs):

```yaml
wandb:
    dry_run: True                                 # Wandb dry-run (not logging)
    name: ''                                      # Wandb run name
    project: os.environ.get("WANDB_PROJECT", "")  # Wandb project
    entity: os.environ.get("WANDB_ENTITY", "")    # Wandb entity
    tags: []                                      # Wandb tags
    dir: ''                                       # Wandb save folder
checkpoint:
    s3_path: ''       # s3 path for AWS model syncing
    s3_frequency: 1   # How often to s3 sync
```

If you encounter out of memory issues, try a lower `batch_size` parameter in the config file.

NB: if you would rather not use docker, you could create a [conda](https://docs.conda.io/en/latest/) environment via following the steps in the Dockerfile and mixing `conda` and `pip` at your own risks...
### Cuda 11

- If you must use a CUDA11, you should install this version of torch and torchvision.

```
pip3 install --pre torch  -f https://download.pytorch.org/whl/nightly/cu111/torch-1.11.0.dev20211017%2Bcu111-cp36-cp36m-linux_x86_64.whl -U
pip3 install --pre torchvision -f https://download.pytorch.org/whl/nightly/cu111/torchvision-0.12.0.dev20211017%2Bcu111-cp36-cp36m-linux_x86_64.whl -U

```
### horovod
- After installing the torch or making docker-container, you must install horovod.
```
pip install horovod
```

## Datasets

Datasets are assumed to be downloaded in `/data/datasets/<dataset-name>` (can be a symbolic link).

### Dense Depth for Autonomous Driving (DDAD)

**Dense Depth for Automated Driving** ([DDAD](https://github.com/TRI-ML/DDAD)): a new dataset that leverages diverse logs from TRI's fleet of well-calibrated self-driving cars equipped with cameras and high-accuracy long-range LiDARs.  Compared to existing benchmarks, DDAD enables much more accurate 360 degree depth evaluation at range, see the official [DDAD repository](https://github.com/TRI-ML/DDAD) for more info and instructions. You can also download DDAD directly via:

```bash
curl -s https://tri-ml-public.s3.amazonaws.com/github/DDAD/datasets/DDAD.tar | tar -xv -C /data/datasets/
```

### KITTI

The KITTI (raw) dataset used in our experiments can be downloaded from the [KITTI website](http://www.cvlibs.net/datasets/kitti/raw_data.php).
For convenience, we provide the standard splits used for training and evaluation: [eigen_zhou](https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/splits/KITTI/eigen_zhou_files.txt), [eigen_train](https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/splits/KITTI/eigen_train_files.txt), [eigen_val](https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/splits/KITTI/eigen_val_files.txt) and [eigen_test](https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/splits/KITTI/eigen_test_files.txt), as well as pre-computed ground-truth depth maps: [original](https://drive.google.com/file/d/1a6cE0R-k_ljck_7Ic1bNAc1zFiaSqles/view?usp=sharing) and [improved](https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/depth_maps/KITTI_raw_groundtruth.tar.gz).
The full KITTI_raw dataset, as used in our experiments, can be directly downloaded [here](https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/datasets/KITTI_raw.tar.gz) or with the following command:

```bash
# KITTI_raw
curl -s https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/datasets/KITTI_raw.tar | tar -xv -C /data/datasets/
```
## Swin Transformer(Pretrained-model)

- You can download pretrained weight in this [github](https://github.com/microsoft/Swin-Transformer).


## Training

Any training, including fine-tuning, can be done by passing either a `.yaml` config file or a `.ckpt` model checkpoint to [scripts/train.py](./scripts/train.py):

```bash
python3 scripts/train.py <config.yaml or checkpoint.ckpt>
```

If you pass a config file, training will start from scratch using the parameters in that config file. Example config files are in [configs](./configs).
If you pass instead a `.ckpt` file, training will continue from the current checkpoint state.

Note that it is also possible to define checkpoints within the config file itself. These can be done either individually for the depth and/or pose networks or by defining a checkpoint to the model itself, which includes all sub-networks (setting the model checkpoint will overwrite depth and pose checkpoints). In this case, a new training session will start and the networks will be initialized with the model state in the `.ckpt` file(s). Below we provide the locations in the config file where these checkpoints are defined:

```yaml
checkpoint:
    # Folder where .ckpt files will be saved during training
    filepath: /path/to/where/checkpoints/will/be/saved
model:
    # Checkpoint for the model (depth + pose)
    checkpoint_path: /path/to/model.ckpt
    depth_net:
        # Checkpoint for the depth network
        checkpoint_path: /path/to/depth_net.ckpt
    pose_net:
        # Checkpoint for the pose network
        checkpoint_path: /path/to/pose_net.ckpt
```

Every aspect of the training configuration can be controlled by modifying the yaml config file. This include the model configuration (self-supervised, semi-supervised, loss parameters, etc), depth and pose networks configuration (choice of architecture and different parameters), optimizers and schedulers (learning rates, weight decay, etc), datasets (name, splits, depth types, etc) and much more. For a comprehensive list please refer to [configs/default_config.py](./configs/default_config.py).

## Evaluation

Similar to the training case, to evaluate a trained model you need to provide a `.ckpt` checkpoint, followed optionally by a `.yaml` config file that overrides the configuration stored in the checkpoint.

```bash
python3 scripts/eval.py --checkpoint <checkpoint.ckpt> [--config <config.yaml>]
```

You can also directly run inference on a single image or folder:

```bash
python3 scripts/infer.py --checkpoint <checkpoint.ckpt> --input <image or folder> --output <image or folder> [--image_shape <input shape (h,w)>]
```

## Models

### DDAD

| Model | Abs.Rel. | Sqr.Rel | RMSE | RMSElog | d < 1.25 |
| :--- | :---: | :---: | :---: |  :---: |  :---: |
| [ResNet18, Self-Supervised, 384x640, ImageNet &rightarrow; DDAD (D)](https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/models/ResNet18_MR_selfsup_D.ckpt)* | 0.227 | 11.293 | 17.368 | 0.303 | 0.758 |
| [PackNet,  Self-Supervised, 384x640, DDAD (D)](https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/models/PackNet01_MR_selfsup_D.ckpt)* | 0.173 | 7.164 | 14.363 | 0.249 | 0.835 |
| [TransDSSL,  Velself-Supervised, 384x640, DDAD (D)](https://drive.google.com/file/d/16Dqxk2c2Uq00ieJOL_XBQIDve_S4Z0Uh/view?usp=sharing)* | 0.151 | 3.591 | 14.350 | 0.244 | 0.815 |

### KITTI

| Model | Abs.Rel. | Sqr.Rel | RMSE | RMSElog | d < 1.25 |
| :--- | :---: | :---: | :---: |  :---: |  :---: |
| [ResNet18, Self-Supervised, 192x640, ImageNet &rightarrow; KITTI (K)](https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/models/ResNet18_MR_selfsup_K.ckpt) | 0.116 | 0.811 | 4.902 | 0.198 | 0.865 |
| [PackNet, Self-Supervised, 192x640, KITTI (K)](https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/models/PackNet01_MR_selfsup_K.ckpt) | 0.111 | 0.800 | 4.576 | 0.189 | 0.880 |
| [TransDSSL, VelSelf-Supervised, 192x640, KITTI (K)](https://drive.google.com/file/d/1mRmmm9Inifuq9Wod3TXsmTN9Ou6ZdzPv/view?usp=sharing) | 0.098 | 0.728 | 4.458 | 0.176 | 0.892 |

All experiments followed the [Eigen et al.](https://arxiv.org/abs/1406.2283), with [Zhou et al](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/)'s preprocessing to remove static training frames. 


## References

Our code is based the [**PackNet**](#cvpr-packnet).

<a id="cvpr-packnet"> </a>
**3D Packing for Self-Supervised Monocular Depth Estimation (CVPR 2020 oral)** \
*Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos and Adrien Gaidon*, [**[paper]**](https://arxiv.org/abs/1905.02693), [**[video]**](https://www.youtube.com/watch?v=b62iDkLgGSI)

```
@inproceedings{packnet,
  author = {Vitor Guizilini and Rares Ambrus and Sudeep Pillai and Allan Raventos and Adrien Gaidon},
  title = {3D Packing for Self-Supervised Monocular Depth Estimation},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  primaryClass = {cs.CV}
  year = {2020},
}
```
