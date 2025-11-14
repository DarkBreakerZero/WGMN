# WGMN:Wavelet-Guided Mamba Network for Texture-Faithful CT Super-Resolution
Tong Lin, Yikun Zhang, Yang Chen, et al.


## Abstract
> Thin-slice computed tomography (CT) is critical for fine anatomical assessment but remains inaccessible in many clinical settings due to hardware and dose constraints. This work proposes two-stage traning process for texture-faithful CT super-resolution (SR), which integrates a structural super-resolution stage and a textural enhancement stage to progressively restore fine anatomical structures and realistic textures. In the structural super-resolution stage, a stationary wavelet transform (SWT)-based fidelity term is incorporated into the training objective to maintain frequency consistency across multiple scales between the SR output and high-resolution (HR) CT, thereby enhancing anatomical structures restoration. In the textural enhancement stage, a Texture Preserver is designed to mitigate texture degradation in the coronal and sagittal views, caused by the anisotropic nature of 3D CT volumes.  To overcome the limited ability of convolutional neural networks (CNNs) in capturing long-range dependencies, the proposed Wavelet-Guided Mamba Network (WGMN) employs Sliding-Window Mamba modules with multi-view feature permutation, enabling efficient modeling of volumetric contextual information. Experiments on three public CT datasets (RPLHR-CT, MSD, Auto-PET) demonstrate that WGMN achieves superior performance in restoring fine structural details and enhancing realistic textures compared with existing methods. Ablation studies further verify that the SWT loss function significantly maintains high-frequency consistency between SR output and HRCT, while the Texture Preserver enhances texture transfer compared to the baseline model. Moreover, WGMN exhibits significant advantages in inference time and memory efficiency over Transformer-based counterparts.

## Architecture Diagram
<p align="center"> <img src="./assets/network.png" width="100%"> </p>

## Visual Comparisons
<p align="center"> <img src="./assets/Comparison.png" width="100%"> </p>


## Usage
### 1.Environment Setup
```
cd code
```
### 2. Model Training
To begin Stage I training:
```
python train_stage1.py --path_key HD --gpu_idx 0 --model mambav2 --net_idx HD_Mamba
```

To begin Stage II training:

```
python train_stage2.py
```

### 3. Model test

To begin Stage I testing:
```
python test_stage1.py --path_key HD --gpu_idx 0 --model mambav2 --net_idx HD_Mamba
```

To begin Stage II testing:
```
python test_stage2.py
```

## Pretrained Weights
Pretrained MBSRN and Texture-Preserver can be downloaded here: [baidu cloud disk](https://pan.baidu.com/s/1cKZYfcVtFJ1dp8--i-fv_A?pwd=p9st).  


## Dataset Structure

The dataset directory of Stage I should be organized as follows:
```
data/
|-- HD_1mm/
|  |-- CT00000000.nii.gz
|  |-- CT00000001.nii.gz
|  |-- CT00000002.nii.gz
|-- HD_5mm/
|  |-- CT00000000.nii.gz
|  |-- CT00000001.nii.gz
|  |-- CT00000002.nii.gz
```
HD_1mm/: High-resolution thin-slice CT data

HD_5mm/: Corresponding thick-slice CT data for training input

The dataset directory of Stage II should be organized as follows:

```
data/
|-- train/
|  |-- full/
|       |-- full-0001.npy
|       |-- full-0002.npy
|       |-- full-0003.npy
|  |-- sparse/
|       |-- sparse-0001.npy
|       |-- sparse-0002.npy
|       |-- sparse-0003.npy
|-- test/
|  |-- full/
|       |-- full-0001.npy
|       |-- full-0002.npy
|       |-- full-0003.npy
|  |-- sparse/
|       |-- sparse-0001.npy
|       |-- sparse-0002.npy
|       |-- sparse-0003.npy
```
## Notes
Code and model configurations are tested under Python 3.9+, PyTorch 2.2+ and Mamba-ssm 2.2.4 environments


## Acknowledgment

This repository is an implementation and extended adaptation of WGMN:Wavelet-Guided Mamba Network for
 Texture-Faithful CT Super-Resolution, with additional refinements in code structure, documentation, and model reproducibility.


