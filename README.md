# Retinal Vessel Segmentation using Deep Learning

## Overview

This project implements advanced deep learning architectures for automated segmentation of retinal vessels and membranes in OCT (Optical Coherence Tomography) images. The system leverages multiple U-Net variants to achieve precise medical image segmentation for clinical applications.

![Processed Image and Label](/assets/Processed%20Image%20and%20Label.png)
*Processed OCT image (left) and corresponding segmentation mask (right) from our ERM dataset, demonstrating the input-output relationship of our segmentation pipeline*

## Project Purpose

This research project focuses on developing robust automated solutions for:

- **Retinal Vessel Segmentation**: Precise identification and delineation of blood vessels in retinal images
- **Epiretinal Membrane Detection**: Automated detection and segmentation of pathological membranes
- **Clinical Decision Support**: Providing quantitative analysis tools for ophthalmologists
- **Medical Image Analysis**: Advancing computer vision techniques in healthcare applications

## Technologies & Methodologies

### Deep Learning Architectures

- **U-Net (Standard)**: Baseline architecture for medical image segmentation
- **Nested U-Net (U-Net++)**: Enhanced skip connections for improved feature propagation
- **Attention U-Net**: Attention mechanisms for focused feature learning
- **Spatial Attention U-Net**: Advanced spatial attention for precise localization
- **ResNet U-Net**: Residual connections for deeper network training

### Technical Stack

- **Framework**: PyTorch for deep learning implementation
- **Image Processing**: Advanced preprocessing pipelines for OCT data
- **Loss Functions**: Dice Loss, Combined Loss functions optimized for medical segmentation
- **Evaluation Metrics**: Comprehensive medical imaging evaluation including Dice coefficient, IoU, sensitivity, and specificity
- **Data Augmentation**: Specialized augmentation techniques for medical imaging

### Datasets

- **ERM Dataset**: Epiretinal membrane segmentation with expert annotations
- **Retina_all**: Comprehensive retinal structure segmentation
- **Vesicles_binary**: Binary vessel segmentation for vascular analysis

## Key Features

✅ **Multi-Architecture Support**: Implementation of 5 different U-Net variants  
✅ **Medical Image Optimization**: Specialized preprocessing for OCT imaging  
✅ **Robust Evaluation**: Comprehensive metrics tailored for medical segmentation  
✅ **Flexible Training**: Support for different loss functions and optimization strategies  
✅ **Clinical Validation**: Evaluation against expert-annotated ground truth  

## Applications

- **Diabetic Retinopathy Screening**: Early detection of vascular changes
- **Epiretinal Membrane Analysis**: Quantitative assessment of membrane thickness and extent
- **Surgical Planning**: Pre-operative visualization and planning tools
- **Disease Progression Monitoring**: Longitudinal analysis of retinal changes
- **Research & Development**: Platform for advancing medical AI applications

## Research Impact

This project contributes to the advancement of computer-aided diagnosis in ophthalmology by providing:
- State-of-the-art segmentation accuracy for retinal structures
- Comparative analysis of different deep learning architectures
- Robust evaluation framework for medical image segmentation
- Clinical workflow integration capabilities

## Quick Start

```bash
# Training Example
cd scripts
python train.py --db_name ERM_W --root ../datasets --total_epochs 200

# Testing Example  
python test.py --db_name ERM_W --root ../datasets --model_path ../outputs/models/best_model.pt

# Evaluation
python evaluate.py --model_path ../outputs/models/best_model.pt --test_data ../datasets/ERM_W/test
```

## Important Note

**🔒 Code Privacy & Security**: This repository contains a curated subset of our research codebase. For organizational and security reasons, certain proprietary algorithms, advanced preprocessing techniques, and specialized model implementations are intentionally kept private. The published code represents the core functionality while maintaining the confidentiality of sensitive research components and institutional intellectual property.

## Results & Performance

The image above demonstrates our preprocessing pipeline's capability to generate high-quality image-mask pairs essential for training robust segmentation models. Our multi-architecture approach enables comprehensive comparison and selection of optimal models for specific clinical applications.

---

*This project represents ongoing research in medical image analysis and computer-aided diagnosis. For collaboration opportunities or research inquiries, please contact.*