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

## 3. Requirements / Dependencies
说明 Python、PyTorch、CUDA、MMCV、MMPose 等环境版本。

## 4. Installation
说明 conda 环境创建、依赖安装、编译安装等。

## 5. Dataset Preparation
说明 EgoWholeBody / SceneEgo 等数据集如何下载、如何放置、如何预处理。

## 6. Pretrained Weights / Checkpoints
说明是否提供预训练权重。  
如果不给，可以写：
`Trained model weights are not included at this stage.`

## 7. Training
给训练命令。

## 8. Evaluation
给测试命令，以及如何得到 MPJPE / PA-MPJPE。

## 9. Citation
给 BibTeX 引用格式。

## 10. Acknowledgements
感谢 EgoWholeBody、SceneEgo、MMPose、相关开源代码等。

