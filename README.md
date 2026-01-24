# PorePatch

**Official implementation and dataset for the paper: "Pore-scale Image Patch Dataset and A Comparative Evaluation of Pore-scale Facial Features"**

> **Authors:** Hualiang Lin, Dong Li*, JiaYu Li  
> *School of Automation, Guangdong University of Technology*  
> *(* Corresponding Author)*

## 📖 Introduction

**PorePatch** is a large-scale, high-quality image patch dataset specifically designed for learning local descriptors on **weak-textured facial skin regions**. It is constructed using a novel **Data-Model Co-Evolution (DMCE)** framework from high-resolution multi-view facial images.

This repository contains the dataset download links and instructions for reproducing the experiments (Patch Verification and 3D Face Reconstruction) presented in the paper.

---

## 📂 Dataset Download

The PorePatch dataset is organized into three parts. Please download the specific files required for your task.

**🔗 Download Link:** [ **https://pan.baidu.com/s/15oQjBbVS-U-lMYxXhYlHFQ?pwd=me5h** ]

### File Descriptions

| File Name | Content Description | Target Task |
| :--- | :--- | :--- |
| **`[PorePatch_Train.zip]`** | The core training dataset containing pore-scale facial patches. | **Training** |
| **`[PorePatch_Verification_Test.zip]`** | The test set structured for standard patch verification (comparable to UBC PhotoTour). | **Patch Verification** |
| **`[PorePatch_Reconstruction_Test.zip]`** | Raw multi-view images of held-out subjects for downstream evaluation. | **3D Face Reconstruction** |

---

## 🚀 Usage

### 1. Training & Patch Verification

For training descriptors and evaluating patch verification performance (FPR95), our pipeline is built upon the **HyNet** framework.

**Reference Repository:** [https://github.com/yuruntian/HyNet](https://github.com/yuruntian/HyNet)

#### Instructions:
1.  **Clone the HyNet repository:**
    ```bash
    git clone https://github.com/yuruntian/HyNet.git
    cd HyNet
    ```
2.  **Prepare Data:**
    *   Download and unzip **`[PorePatch_Train.zip]`** for training.
    *   Download and unzip **`[PorePatch_Verification_Test.zip]`** for evaluation.
    *   Place them in the data directory expected by HyNet code.
3.  **Modify Model:**
    *   Replace the model definition in `model.py` (or equivalent) with the descriptor architecture you wish to test (e.g., your own model, or the AFSRNet/SDGMNet mentioned in our paper).
4.  **Run:**
    *   Follow the original HyNet instructions to start training or testing.

### 2. 3D Face Reconstruction (Downstream Task)

To evaluate the performance of descriptors in the real-world geometric task (3D Reconstruction), we follow the **ETH Local Feature Evaluation Benchmark** protocol.

**Reference Repository:** [https://github.com/ahojnnes/local-feature-evaluation](https://github.com/ahojnnes/local-feature-evaluation)

#### Instructions:
1.  **Clone the Evaluation repository:**
    ```bash
    git clone https://github.com/ahojnnes/local-feature-evaluation.git
    ```
2.  **Prepare Data:**
    *   Download **`[PorePatch_Reconstruction_Test.zip]`**.
    *   Replace the default ETH benchmark datasets with our provided **Reconstruction Test Set**.
3.  **Run Pipeline:**
    *   Extract features using your trained descriptor model.
    *   Run the COLMAP reconstruction pipeline provided in the repository to generate sparse and dense point clouds.
    *   Evaluate metrics such as **Dense Points**, **Sparse Points**, and **Reprojection Error**.

---

## 📝 Citation

If you use the PorePatch dataset or this code in your research, please cite our paper:

```bibtex
@article{li2025pore,
  title={Pore-scale Image Patch Dataset and A Comparative Evaluation of Pore-scale Facial Features},
  author={Li, Dong and Lin, HuaLiang and Li, JiaYu},
  journal={arXiv preprint arXiv:2512.00381},
  year={2025}
}
