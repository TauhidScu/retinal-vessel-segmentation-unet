# Advanced OCT Segmentation Framework

<div align="center">

![OCT Segmentation Banner](/assets/banner.png)

[![Python](https://img.shields.io/badge/Python-3.7.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/Version-1.0-brightgreen.svg)](README.md)

*Advanced deep learning framework for automated OCT image segmentation and analysis. This comprehensive framework addresses critical challenges in retinal imaging through state-of-the-art neural architectures, enabling precise detection of epiretinal membranes, retinal vessel segmentation, and multi-class vesicle classification for enhanced clinical diagnosis and treatment planning.*

</div>

---

## 📋 Table of Contents

- [Research Background](#research-background)
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Dataset Information](#dataset-information)
- [Model Architectures](#model-architectures)
- [Results & Performance](#results--performance)
- [Clinical Applications](#clinical-applications)
- [Code Privacy & Security](#code-privacy--security)
- [Contributing](#contributing)
- [References](#references)

---

## Research Background

This project represents a **collaborative research initiative** between the **Machine Intelligence Lab and  Department of Ophthalmology**, Sichuan University focusing on advanced deep learning solutions for OCT image analysis.
It mainly includes data collection, data aannotation, data processing, model design and validation where data collection and data annotation is handaled by the expert from west school of Ophthalmology. 

### **Research Objectives**

Our comprehensive framework addresses critical challenges in ophthalmology through:

- **🔍 Epiretinal Membrane Segmentation**: Automated segmentation and quantification of pathological membranes  
- **👁️ Retina segmentation**: Complete anatomical segmentation for clinical assessment
- **🧬 Vesicles(Cyst)**: Binary and Multi-class segmentation of vesicles components distributed in different layers


## Key Features

### 🎯 **Multi-Task Segmentation Capabilities**

- **Binary Segmentation**: ERM detection and retinal boundary delineation
- **Multi-class Segmentation**: Vesicle classification (Background, Layer 2, Layer 4)
- **Vascular Analysis**: Comprehensive vessel segmentation and quantification

### 🧠 **Advanced Deep Learning Architectures**

- **U-Net (Standard)**: Robust baseline for medical image segmentation [[1]](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
        
        
- **Nested U-Net (U-Net++)**: Enhanced feature propagation through dense skip connections [[2]](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1)
        
        
- **Attention U-Net**: Focused feature learning with attention mechanisms [[3]](https://doi.org/10.48550/arXiv.1804.03999)
        
        
- **Spatial Attention U-Net (SAUNet)**: Advanced spatial attention for precise localization [[4]](https://ieeexplore.ieee.org/document/9413346)
- **ResNet U-Net**: Residual connections enabling deeper network training [[5]](https://ieeexplore.ieee.org/document/7780459)

### 📊 **Comprehensive Evaluation Framework**

- **Clinical Metrics**: Sensitivity, specificity, accuracy tailored for medical applications
- **Segmentation Quality**: Dice coefficient, IoU, Hausdorff distance
- **Statistical Analysis**: Confidence intervals and significance testing
- **Cross-validation**: Robust performance assessment across multiple folds

---

## Architecture Overview

### **Multi-Model Framework**

```mermaid
graph TD
    A[OCT Input Image] --> B[Preprocessing Pipeline]
    B --> C{Model Selection}
    C --> D[U-Net Standard]
    C --> E[Nested U-Net]
    C --> F[Attention U-Net]
    C --> G[SAUNet]
    C --> H[ResNet U-Net]
    D --> I[Ensemble Prediction]
    E --> I
    F --> I
    G --> I
    H --> I
    I --> J[Post-processing]
    J --> K[Clinical Output]
```


## Quick Start

### **Training Example**

```bash
# Navigate to scripts directory
cd scripts

# Train ERM segmentation model
python train.py --db_name ERM_W --root ../datasets --total_epochs 200 --model_type attention_unet

# Train retinal segmentation
python train.py --db_name Retina_all --root ../datasets --total_epochs 150 --model_type nested_unet

# Train vesicle classification
python train.py --db_name Vesicles_binary --root ../datasets --total_epochs 180 --model_type unet
```

### **Testing & Evaluation**

```bash
# Test trained model
python test.py --db_name ERM_W --root ../datasets --model_path ../outputs/models/best_model.pt

# Comprehensive evaluation
python evaluate.py --model_path ../outputs/models/best_model.pt --test_data ../datasets/ERM_W/test
```

### **Quick Inference**

```bash
# Single image inference
python inference.py --image_path sample.jpg --model_type attention_unet --output_dir results/
```

---

## Installation

### **System Requirements**

| Component | Specification |
|-----------|---------------|
| **Python** | 3.7.12+ |
| **GPU** | NVIDIA RTX-A4000 (recommended) |
| **Memory** | 16GB+ RAM |
| **Storage** | 50GB+ available space |

### **Installation Steps**

```bash
# Clone repository
git clone https://github.com/TauhidScu/Advanced-OCT-Segmentation.git
cd Advanced-OCT-Segmentation

# Create conda environment
conda create -n oct-seg python=3.7.12
conda activate oct-seg

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

---

## Dataset Information

The dataset was collected following standard regulation and the dataset is private.

| Dataset | Total Images | Resolution | Task Type | Clinical Focus |
|---------|-------------|------------|-----------|----------------|
| **ERM** | 363 | 1536×1024 | Binary Segmentation | Epiretinal Membrane Detection |
| **Retina** | 663 | 1536×1024 | Binary Segmentation | Complete Retinal Boundary |
| **Vesicles** | 694 | 1536×1024 | Multi-class | Layer 2 & 4 Vesicle Classification |


## Model Architectures
Al the models were implemented following the original paper.
### **Implemented Models**

#### **1. U-Net (Baseline)** [[1]](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)

- **Architecture**: Standard encoder-decoder with skip connections
- **Strengths**: Robust performance, well-established baseline
- **Best for**: General segmentation tasks

#### **2. Nested U-Net (U-Net++)** [[2]](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1)

- **Architecture**: Dense skip connections between encoder-decoder paths
- **Strengths**: Enhanced feature reuse, improved gradient flow
- **Best for**: Complex boundary detection

#### **3. Attention U-Net** [[3]](https://doi.org/10.48550/arXiv.1804.03999)

- **Architecture**: Attention gates integrated into U-Net structure
- **Strengths**: Focused learning on relevant regions
- **Best for**: Noisy images with subtle pathologies

#### **4. Spatial Attention U-Net (SAUNet)** [[4]](https://ieeexplore.ieee.org/document/9413346)

- **Architecture**: Spatial attention gates with DropBlock regularization
- **Strengths**: Precise spatial localization, overfitting prevention
- **Best for**: High-precision segmentation tasks

#### **5. ResNet U-Net** [[5]](https://ieeexplore.ieee.org/document/7780459)

- **Architecture**: ResNet18 encoder with U-Net decoder
- **Strengths**: Deep feature extraction, pre-trained weights
- **Best for**: Transfer learning applications

---

## Results & Performance

### **Qualitative Results Comparison**

![Model Comparison Results](/assets/model_comparison_results.jpg)

*Qualitative comparison of segmentation results across all implemented models. From top to bottom: Original OCT images, U-Net, Attention U-Net, U-Net++, SAUNet, ResNet U-Net, and Ground Truth (GT). The comparison demonstrates each model's ability to accurately delineate retinal boundaries, with visual assessment showing consistent performance across different architectures while highlighting subtle differences in boundary precision and noise handling.*

### **Retina segmentation Performance**

| Model | F1 Score | Sensitivity | Specificity | Precision | IoU | Accuracy |
|-------|----------|-------------|-------------|-----------|-----|----------|
| **U-Net** | **0.9706** | 0.9689 | 0.9965 | **0.9723** | **0.943** | **0.9934** |
| **U-Net++** | 0.9661 | **0.9798** | 0.9938 | 0.9533 | 0.9349 | 0.9922 |
| **Attention U-Net** | 0.9526 | 0.9849 | 0.9887 | 0.9245 | 0.9116 | 0.9883 |
| **SAUNet** | 0.9524 | 0.9744 | 0.9904 | 0.9325 | 0.9105 | 0.9886 |
| **ResNet U-Net** | 0.9319 | 0.9834 | **0.983** | 0.8904 | 0.8763 | 0.983 |

### **Multi-class vesicles(cysts) segmentation Performance**

| Model | F1 Score | Sensitivity | Specificity | Precision | IoU | Accuracy |
|-------|----------|-------------|-------------|-----------|-----|----------|
| **U-Net** | **0.8177** | **0.8268** | 0.9992 | 0.8177 | **0.693** | 0.9985 |
| **U-Net++** | 0.8122 | 0.8048 | **0.9993** | **0.8287** | 0.6845 | 0.9985 |
| **Attention U-Net** | 0.8144 | 0.7778 | **0.9995** | 0.8645 | 0.6881 | **0.9986** |
| **SAUNet** | 0.7811 | 0.7237 | **0.9995** | 0.8532 | 0.6411 | 0.9983 |
| **ResNet U-Net** | 0.7769 | 0.7394 | **0.9993** | 0.8322 | 0.6354 | 0.9983 |

### **Binary Vesicles segmentation Performance**

| Model | F1 Score | Sensitivity | Specificity | Precision | IoU | Accuracy |
|-------|----------|-------------|-------------|-----------|-----|----------|
| **U-Net** | 0.6991 | 0.7265 | **0.9997** | 0.6996 | 0.5423 | 0.9994 |
| **U-Net++** | **0.7115** | **0.75** | **0.9997** | 0.6961 | **0.555** | **0.9995** |
| **Attention U-Net** | 0.7538 | 0.7396 | **0.9998** | **0.7852** | 0.6098 | **0.9995** |
| **SAUNet** | 0.6813 | 0.7548 | **0.9997** | 0.7164 | 0.5752 | **0.9995** |
| **ResNet U-Net** | 0.6493 | 0.7033 | 0.9996 | 0.6246 | 0.488 | 0.9993 |

### **Performance Insights**

#### **Retina Dataset Analysis:**

- **Outstanding Performance**: All models achieve >93% F1 scores, demonstrating excellent retinal boundary detection
- **Best Overall**: U-Net achieves the highest F1 score (0.9706) and IoU (0.943)
- **High Sensitivity**: U-Net++ shows the highest sensitivity (0.9798) for detecting retinal structures
- **Exceptional Accuracy**: All models exceed 98% accuracy, indicating robust clinical applicability

#### **Vesicles Dataset Analysis:**

- **Consistent Performance**: All models achieve >0.77 F1 scores, indicating robust architecture
- **Excellent Specificity**: Most models achieve >0.999 specificity, crucial for clinical applications
- **High Accuracy**: All models exceed 99.8% accuracy, demonstrating reliable performance
- **Best Balance**: U-Net provides optimal balance between sensitivity and precision

#### **Vesicles Binary Classification Analysis:**

- **Attention Mechanism Advantage**: Attention U-Net shows superior precision (0.7852) for binary classification
- **Robust Specificity**: All models maintain >99.9% specificity, minimizing false positives
- **Clinical Reliability**: Consistent high accuracy across all architectures ensures clinical viability

### **Qualitative Analysis**

The visual comparison reveals several key insights:

- **Boundary Precision**: All models demonstrate excellent boundary delineation capabilities
- **Noise Handling**: Attention mechanisms (AttUNet, SAUNet) show superior noise suppression
- **Consistency**: Results show remarkable consistency across different retinal structures
- **Clinical Viability**: Visual outputs closely match ground truth annotations, supporting clinical deployment

---

## Clinical Applications

### **Diagnostic Applications**

1. **Diabetic Retinopathy Screening**
   - Early detection of vascular changes
   - Automated severity grading
   - Population-level screening programs

2. **Epiretinal Membrane Analysis**
   - Quantitative assessment of membrane thickness
   - Surgical planning support
   - Treatment response monitoring

3. **Disease Progression Monitoring**
   - Longitudinal analysis capabilities
   - Quantitative biomarker extraction
   - Treatment efficacy assessment

4. **Retinal Layer Analysis**
   - Precise vesicle classification
   - Layer thickness measurements
   - Structural integrity assessment


## Code Privacy & Security

**🔒 Important Notice**: This repository contains a **curated subset** of our research codebase developed through the collaboration between **Machine Intelligence Lab** and **Sichuan University**. 

For **organizational security** and **intellectual property protection**, certain components are intentionally kept private:

- **Proprietary Algorithms**: Advanced preprocessing techniques and specialized optimization methods
- **Clinical Integration**: Hospital-specific workflow adaptations and DICOM processing modules  
- **Advanced Models**: Cutting-edge architectures under development and patent review
- **Institutional Data**: Sensitive datasets and patient information handling protocols
- **Performance Optimizations**: Hardware-specific acceleration and deployment strategies

The **published code represents core functionality** while maintaining confidentiality of sensitive research components and institutional intellectual property. This approach ensures:

- **Open Science**: Sharing fundamental research contributions with the community
- **Security Compliance**: Protecting sensitive medical data and proprietary methods
- **Collaboration**: Enabling external researchers to build upon our work
- **Innovation Protection**: Safeguarding competitive advantages and ongoing research

For **full access** or **research collaboration** opportunities, please contact us.

---

## Contributing

We welcome contributions from the global research community! Here's how you can participate:

### **Ways to Contribute**

1. **Bug Reports**: Help us identify and fix issues
2. **Feature Requests**: Suggest improvements and new capabilities
3. **Research Contributions**: Share your findings and model improvements
4. **Clinical Validation**: Provide clinical feedback and validation data
5. **Documentation**: Improve tutorials and documentation

### **Research Collaboration**

We actively seek partnerships with:

- **Academic Institutions** for joint research projects
- **Clinical Centers** for validation studies
- **Technology Companies** for deployment and scaling
---

## References

[1] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015*, 234-241. https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

[2] Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018). UNet++: A Nested U-Net Architecture for Medical Image Segmentation. *Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support*, 3-11. https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1

[3] Oktay, O., Schlemper, J., Folgoc, L. L., Lee, M., Heinrich, M., Misawa, K., ... & Rueckert, D. (2018). Attention U-Net: Learning Where to Look for the Pancreas. *arXiv preprint arXiv:1804.03999*. https://doi.org/10.48550/arXiv.1804.03999

[4] Guo, C., Szemenyei, M., Yi, Y., Wang, W., Chen, B., & Fan, C. (2021). SA-UNet: Spatial Attention U-Net for Retinal Vessel Segmentation. *2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC)*. https://ieeexplore.ieee.org/document/9413346

[5] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770-778. doi: 10.1109/CVPR.2016.90

---

<div align="center">

*Advancing Medical AI Through International Collaboration*

</div>

---
