# Structure-Aware-Joint-Limb-Voxel-Pose

## 1. Overview
This repository provides the implementation of **Structure-Aware Joint--Limb Voxel Modelling for Egocentric 3D Human Pose Estimation**.

Existing voxel-based egocentric pose estimation methods can reduce monocular depth ambiguity, but they usually predict joint voxel heatmaps independently and lack explicit modelling of joint--limb structural dependencies in 3D space. Directly modelling such interactions over dense voxel grids is also computationally expensive.

To address this problem, we propose a structure-aware voxel framework that jointly models discrete body joints and continuous limb segments. The method introduces an auxiliary limb voxel branch for bone-level structural supervision, and further uses sparse Top-K voxel tokenisation with CrossTGFI3D to efficiently fuse joint and limb information under topology and geometry constraints.
<p align="center">
  <img src="assets/fig1.png" width="700">
</p>

## 2. Key Algorithms

### Joint--Limb Dual-Branch Voxel Modelling

The framework predicts both joint voxel heatmaps and limb voxel heatmaps. The joint branch localises discrete body keypoints, while the auxiliary limb branch models continuous limb segments in 3D space. This design provides bone-level structural supervision and helps the network preserve anatomically consistent poses.

### Sparse Top-K Voxel Tokenisation

Directly modelling interactions over the full 3D voxel grid is computationally expensive. To address this, the method selects only the Top-K highest-response voxels from each joint and limb heatmap. These selected voxels are converted into compact tokens using their response values and 3D coordinates, enabling efficient structural reasoning over informative regions.
<p align="center">
  <img src="assets/fig2.png" width="500">
</p>

<p align="center">
  <img src="assets/fig4.png" width="500">
</p>

### CrossTGFI3D: Geometry-aware Joint--Limb Fusion

CrossTGFI3D performs cross-branch attention between joint tokens and limb tokens. It uses a geometry-aware distance bias to encourage interactions between spatially close regions and a topology-guided mask to restrict attention to anatomically valid joint--limb connections. The enhanced joint tokens are then written back to the joint heatmaps for final 3D pose prediction.

<p align="center">
  <img src="assets/fig3.png" width="500">
</p>

## 3. Installation

 We recommend creating an isolated conda environment before installing the required packages.

### Create a conda environment

```bash
conda create -n joint_limb_pose python=3.9 -y
conda activate joint_limb_pose
```

### Install PyTorch

Install the PyTorch version that matches your CUDA environment. For example, the experiments in this project were conducted with PyTorch 1.10.1 and CUDA 11.1.

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

### Install MMCV and MMPose-related dependencies

```bash
pip install openmim
mim install mmcv-full==1.7.1
pip install -e .
```

### Install additional dependencies

```bash
pip install -r requirements.txt
```

If smplx installed open3d-python, you should uninstall it by running:

```bash
pip uninstall open3d-python
```

## 4. Dataset Preparation

This project uses the following publicly available egocentric pose estimation datasets.

### EgoWholeBody Training Dataset

The EgoWholeBody training dataset can be downloaded from the official Edmond repository:

https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.SJYBX3

### EgoWholeBody Test Dataset

The EgoWholeBody test dataset can be downloaded from the official Edmond repository:

https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.FSBR5V

### SceneEgo Dataset

The SceneEgo dataset can be downloaded from the official Edmond repository:

https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.VCIHDO

After downloading the datasets, unzip all of the files. please organise them as follows:

```text
data/
├── EgoWholeMocap/
│   ├── train/
│   │   ├── path_to_dataset_dir
│   │   │   ├── renderpeople_adanna
│   │   │   ├── renderpeople_amit
│   │   │   ├── ......
│   │   │   ├── renderpeople_mixamo_labels_old.pkl
│   │   │   └── ......
│   │   └── ......
│   └── test/
│       └──render_people_mixamo_test_seq
│          ├── render_people_manuel
│          ├── ......
│          └── renderpeople_mixamo_labels_test_seq.pkl
│ 
└── SceneEgo/
    ├── train/
    │   ├── diogo1
    │   ├── ......
    │   └── pranay2
    └── test/
        ├── diogo1
        ├── ......
        └── jian2
```

## 5. Pretrained Weights / Checkpoints
说明是否提供预训练权重。  
如果不给，可以写：
`Trained model weights are not included at this stage.`

## 6. Training
给训练命令。

## 7. Evaluation
给测试命令，以及如何得到 MPJPE / PA-MPJPE。

## 8. Citation
给 BibTeX 引用格式。

## 9. Acknowledgements
感谢 EgoWholeBody、SceneEgo、MMPose、相关开源代码等。

