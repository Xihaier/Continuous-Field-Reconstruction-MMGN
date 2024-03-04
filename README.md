<div align="center">

# Continuous Field Reconstruction from Sparse Observations with Implicit Neural Networks (ICLR 2024)

<a href="https://pytorch.org/get-started/locally/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch -ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-89b8cd?style=for-the-badge&labelColor=gray"></a>

<h3> ✨Official implementation of our <a href="https://openreview.net/forum?id=kuTZMZdCPZ">MMGN</a> model✨ </h3>
 
</div>

## Abstract

Reliably reconstructing physical fields from sparse sensor data is a challenge that frequently arises in many scientific domains. In practice, the process generating the data often is not understood to sufficient accuracy. Therefore, there is a growing interest in using the deep neural network route to address the problem. This work presents a novel approach that learns a continuous representation of the physical field using implicit neural representations (INRs). Specifically, after factorizing spatiotemporal variability into spatial and temporal components using the separation of variables technique, the method learns relevant basis functions from sparsely sampled irregular data points to develop a continuous representation of the data. In experimental evaluations, the proposed model outperforms recent INR methods, offering superior reconstruction quality on simulation data from a state-of-the-art climate model and a second dataset that comprises ultra-high resolution satellite-based sea surface temperature fields.

For more information, please refer to the following:

- Xihaier Luo and Wei Xu and Balu Nadiga and Yihui Ren and Shinjae Yoo. "[Continuous Field Reconstruction from Sparse Observations with Implicit Neural Networks](https://openreview.net/forum?id=kuTZMZdCPZ)." (ICLR 2024).

## Setting it up

First, clone the project.

```bash
# clone project
git clone https://github.com/Xihaier/Continuous-Field-Reconstruction-MMGN.git
cd Continuous-Field-Reconstruction-MMGN
```

Then, install the needed dependencies.

```bash
# install dependencies
conda env create -f environment.yml
```

## Get Started

1. Prepare the Data: To access the experimental datasets, please use the following link. After downloading, unzip the file to reveal two datasets referenced in the paper. These datasets should be placed in the data folder.

| Dataset | Link |
| -------------------------------------------- | ------------------------------------------------------------ |
| Simulation-based global surface temperature | [[Google Cloud]](https://drive.google.com/file/d/1V8DvkcWZrW2Z_4eaz4VjtMU8IvcLyyMe/view) |
| Satellite-based sea surface temperature | [[Google Cloud]](https://drive.google.com/file/d/1V8DvkcWZrW2Z_4eaz4VjtMU8IvcLyyMe/view) |

2. Model Training and Evaluation: To replicate the experiment results, execute the command: `bash run.sh`. Within the `run.sh` file, you have the flexibility to modify the model by selecting from options such as ResMLP, SIREN, FFN_P, FFN_G, and MMGN. Additionally, you can alter the task setting from task1 to task4 and adjust the sampling ratio.

```bash
bash run.sh
```

## Quantitative Results

Performance comparison with four INR baselines on both high-fidelity climate simulation data and real-world satellite-based benchmarks. MSE is recorded. A smaller MSE denotes superior performance. For clarity, we highlight the best result in bold and underline the second-best. We have also included the promotion metric, which indicates the reduction in relative error compared to the second-best model for each task.

<p align="center">
<img src=".\img\main_results.png" height = "350" alt="" align=center />
<br><br>
<b>Table 1.</b> Model performance.
</p>

## Qualitative Results

Visualizations of true and reconstructed fields: (1) global surface temperature derived from multi-scale high-fidelity climate simulations and (2) sea surface temperature assimilated using satellite imagery observations. For each dataset, the first column displays the ground truth, the first row showcases predictions from different models, and the second row presents corresponding error maps relative to the reference data. In the error maps, darker pixels indicate lower error levels.

<p align="center">
<img src=".\img\showcases.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 3.</b> Showcases.
</p>

## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{
luo2024continuous,
title={Continuous Field Reconstruction from Sparse Observations with Implicit Neural Networks},
author={Xihaier Luo and Wei Xu and Balu Nadiga and Yihui Ren and Shinjae Yoo},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=kuTZMZdCPZ}
}
```

## Contact

If you have any questions or want to use the code, please contact [xluo@bnl.gov](mailto:xluo@bnl.gov).