<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1614,50:1c8a5c,100:39ff9d&height=220&section=header&text=Agro-Vision&fontSize=70&fontColor=ffffff&fontAlignY=38&desc=Intelligent%20Plant%20Disease%20Classification%20System&descAlignY=58&descSize=20&animation=fadeIn" width="100%"/>

<a href="https://huggingface.co/spaces/deepak0027/plant-disease-classifier">
  <img src="https://readme-typing-svg.demolab.com/?lines=8+Architectures.+1+Battle.+38+Diseases.;CNNs+vs+Vision+Transformers+%E2%80%94+who+wins%3F;99.13%25+Validation+Accuracy+%F0%9F%8F%86;Built+with+%F0%9F%8C%BF+and+Deep+Learning;&font=JetBrains+Mono&size=20&pause=1500&color=39FF9D&center=true&vCenter=true&width=700&height=45&separator=;" alt="Typing SVG" />
</a>

<br/>

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CV-brightgreen?style=for-the-badge)

<br/>

[![🚀 Live Demo](https://img.shields.io/badge/🌿%20LIVE%20DEMO-Try%20it%20on%20HuggingFace-2d6a4f?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=0d1614)](https://huggingface.co/spaces/deepak0027/plant-disease-classifier)
[![Space Status](https://img.shields.io/badge/Status-Running%20🟢-2d6a4f?style=for-the-badge&labelColor=0d1614)](https://huggingface.co/spaces/deepak0027/plant-disease-classifier)

<br/>

![Best Accuracy](https://img.shields.io/badge/Best%20Val%20Accuracy-99.13%25-39ff9d?style=flat-square&labelColor=0d1614)
![Classes](https://img.shields.io/badge/Disease%20Classes-38-39ff9d?style=flat-square&labelColor=0d1614)
![Species](https://img.shields.io/badge/Crop%20Species-14-39ff9d?style=flat-square&labelColor=0d1614)
![Models](https://img.shields.io/badge/Architectures-8-39ff9d?style=flat-square&labelColor=0d1614)
![Epochs](https://img.shields.io/badge/Epochs%20per%20model-10-39ff9d?style=flat-square&labelColor=0d1614)

<br/>

### *A comprehensive deep learning research project comparing 8 state-of-the-art architectures for automated crop disease detection*

<img src="https://raw.githubusercontent.com/Anmol-Baranwal/Cool-GIFs-For-GitHub/main/leaf-loading.gif" width="0" height="0" alt=""/>

</div>

<div align="center">

```diff
+ Robustness stress-testing   + Grad-CAM explainability   + Live ensemble consensus
+ 8 rival architectures       + Interactive Gradio UI     + One-click dataset gallery
```

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:39ff9d,100:0d1614&height=3&width=100%"/>

## 📑 Table of Contents

<details open>
<summary><b>Click to expand / collapse</b></summary>

- [📌 Project Overview](#-project-overview)
- [🏗️ Architectures Compared](#️-architectures-compared)
- [📊 Dataset](#-dataset)
- [🔬 Methodology](#-methodology)
- [📈 Results Summary](#-results-summary)
- [📁 Project Structure](#-project-structure)
- [🚀 How to Run](#-how-to-run)
- [🛠️ Requirements](#️-requirements)
- [🎮 Interactive Demo](#-interactive-demo)
- [🗺️ Notebook Walkthrough](#️-notebook-walkthrough)
- [💡 Key Insights](#-key-insights)
- [🤝 Acknowledgements](#-acknowledgements)
- [📬 Connect](#-connect)

</details>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:39ff9d,100:0d1614&height=3&width=100%"/>

## 📌 Project Overview

This project builds and evaluates **8 different deep learning models** on the PlantVillage dataset to classify **38 plant disease categories** across multiple crop species. Beyond simple classification, the project includes:

- 🌪️ **Robustness stress testing** using Gaussian noise
- 🔥 **Explainability** via Grad-CAM heatmaps (visualizing what the model "sees")
- 🎮 **Interactive demo dashboards** built with both `ipywidgets` and `Gradio`
- ⚔️ A **head-to-head battle** between CNN-based models and Vision Transformers (ViT)

> **The end goal:** find which architecture delivers the best accuracy, generalization, and real-world robustness for agricultural AI.

<div align="center">

```mermaid
flowchart LR
    A[📸 Upload Leaf] --> B[🧼 Resize + Normalize]
    B --> C{⚙️ Selected Engine}
    C --> D[🧠 Forward Pass]
    D --> E[📊 Softmax + Top-3]
    E --> F[📋 Diagnostic Report]
    style A fill:#0d1614,stroke:#39ff9d,color:#eaf7f0
    style B fill:#0d1614,stroke:#39ff9d,color:#eaf7f0
    style C fill:#12331f,stroke:#39ff9d,color:#39ff9d
    style D fill:#0d1614,stroke:#39ff9d,color:#eaf7f0
    style E fill:#0d1614,stroke:#39ff9d,color:#eaf7f0
    style F fill:#12331f,stroke:#39ff9d,color:#39ff9d
```

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:39ff9d,100:0d1614&height=3&width=100%"/>

## 🏗️ Architectures Compared

<div align="center">

| # | Model | Type | Input Size | Optimizer | LR |
|:-:|-------|------|:-----------:|:-----------:|:-----:|
| 1 | **EfficientNet-B0** | CNN (Baseline) | 224×224 | Adam | `0.001` |
| 2 | **Swin Transformer** | Vision Transformer | 224×224 | Adam | `5e-5` |
| 3 | **ResNet-50** | CNN | 224×224 | Adam | `0.001` |
| 4 | **DenseNet-121** | CNN | 224×224 | Adam | `0.0005` |
| 5 | **Inception-V3** | CNN | 299×299 | Adam | `0.001` |
| 6 | **MobileNet-V3 Large** | Lightweight CNN | 224×224 | Adam | `0.001` |
| 7 | **VGG-16** | Classic Deep CNN | 224×224 | Adam | `0.0001` |
| 8 | **ConvNeXt-Tiny** 🏆 | Modern CNN | 224×224 | AdamW | `0.001` |

</div>

> All models use **Transfer Learning** — pre-trained on ImageNet, with frozen backbones and fine-tuned classification heads.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:39ff9d,100:0d1614&height=3&width=100%"/>

## 📊 Dataset

<div align="center">

**PlantVillage Dataset** — one of the most widely used datasets in agricultural AI research.

| Detail | Info |
|--------|------|
| Source | [Kaggle — PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) |
| Total Classes | **38** (plant–disease combinations) |
| Image Type | Color (RGB) |
| Input Format | `ImageFolder` (PyTorch) |
| Train / Val Split | **80% / 20%** |
| Batch Size | 32 |

</div>

### 🌱 Supported Plant Species

<div align="center">

![Apple](https://img.shields.io/badge/-Apple-1c8a5c?style=flat-square) ![Blueberry](https://img.shields.io/badge/-Blueberry-1c8a5c?style=flat-square) ![Cherry](https://img.shields.io/badge/-Cherry-1c8a5c?style=flat-square) ![Corn](https://img.shields.io/badge/-Corn%20(Maize)-1c8a5c?style=flat-square) ![Grape](https://img.shields.io/badge/-Grape-1c8a5c?style=flat-square) ![Orange](https://img.shields.io/badge/-Orange-1c8a5c?style=flat-square) ![Peach](https://img.shields.io/badge/-Peach-1c8a5c?style=flat-square)

![Bell Pepper](https://img.shields.io/badge/-Bell%20Pepper-1c8a5c?style=flat-square) ![Potato](https://img.shields.io/badge/-Potato-1c8a5c?style=flat-square) ![Raspberry](https://img.shields.io/badge/-Raspberry-1c8a5c?style=flat-square) ![Soybean](https://img.shields.io/badge/-Soybean-1c8a5c?style=flat-square) ![Squash](https://img.shields.io/badge/-Squash-1c8a5c?style=flat-square) ![Strawberry](https://img.shields.io/badge/-Strawberry-1c8a5c?style=flat-square) ![Tomato](https://img.shields.io/badge/-Tomato-1c8a5c?style=flat-square)

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:39ff9d,100:0d1614&height=3&width=100%"/>

## 🔬 Methodology

<details open>
<summary><b>1️⃣ Data Preprocessing</b></summary>

```python
transforms.Resize((224, 224))         # Resize to model input
transforms.ToTensor()                  # Convert to tensor
transforms.Normalize(                  # ImageNet normalization
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```
> Inception-V3 uses `299×299` instead of `224×224`.

</details>

<details>
<summary><b>2️⃣ Transfer Learning Strategy</b></summary>

- Download ImageNet pre-trained weights
- **Freeze** backbone layers (keep learned low-level features)
- **Replace** the final classification head with `nn.Linear(features, 38)`
- Train **only the head** for fast convergence → then optionally unfreeze for full fine-tuning

</details>

<details>
<summary><b>3️⃣ Training Configuration</b></summary>

- **Loss Function:** `CrossEntropyLoss`
- **Epochs:** 10 per model
- **Hardware:** Kaggle GPU (CUDA)
- **History Tracked:** Training loss, Training accuracy, Validation accuracy per epoch

</details>

<details>
<summary><b>4️⃣ Robustness / Stress Test</b></summary>

Each trained model is evaluated under **Gaussian noise** (level = 0.2):

```python
noise = torch.randn_like(images) * 0.2
noisy_images = images + noise
```

This tests whether models generalize to real-world conditions like low-quality camera images, outdoor lighting variation, etc.

</details>

<details>
<summary><b>5️⃣ Explainability — Grad-CAM</b></summary>

Grad-CAM (Gradient-weighted Class Activation Mapping) generates **heatmaps** showing which regions of a leaf image the model focuses on when making predictions.

- **CNN Target Layer:** `cnn_model.features[-1]`
- **ViT Target Layer:** `vit_model.norm` (with reshape transform for Swin)

</details>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:39ff9d,100:0d1614&height=3&width=100%"/>

## 📈 Results Summary

<div align="center">

| Model | Best Val Accuracy | Final Train Loss |
|-------|:-----------------:|:----------------:|
| 🥇 ConvNeXt-Tiny | **99.13%** ![](https://progress-bar.dev/99/?scale=100&width=110&color=39ff9d) | `0.0236` |
| 🥈 Swin Transformer | **98.10%** ![](https://progress-bar.dev/98/?scale=100&width=110&color=39ff9d) | `0.0839` |
| 🥉 VGG-16 | **97.57%** ![](https://progress-bar.dev/98/?scale=100&width=110&color=39ff9d) | `0.0323` |
| ResNet-50 | **97.01%** ![](https://progress-bar.dev/97/?scale=100&width=110&color=39ff9d) | `0.0957` |
| DenseNet-121 | **96.74%** ![](https://progress-bar.dev/97/?scale=100&width=110&color=39ff9d) | `0.2030` |
| MobileNet-V3 | **96.69%** ![](https://progress-bar.dev/97/?scale=100&width=110&color=39ff9d) | `0.0808` |
| EfficientNet-B0 | **96.54%** ![](https://progress-bar.dev/97/?scale=100&width=110&color=39ff9d) | `0.1466` |
| Inception-V3 | **93.60%** ![](https://progress-bar.dev/94/?scale=100&width=110&color=ff5470) | `0.5975` |

</div>

> **🧪 Hypothesis tested:** *Vision Transformers are more robust to image noise than CNNs due to their global attention mechanism.*

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:39ff9d,100:0d1614&height=3&width=100%"/>

## 📁 Project Structure

```
plant-disease-classifier/
│
├── plant-diseases-classification-system.ipynb   # Main notebook (all models)
│
├── saved_models/                                # Trained model weights (.pth)
│   ├── cnn_baseline_plantvillage_10epochs.pth
│   ├── swin_final_plantvillage_10epochs.pth
│   ├── resnet50_plantvillage_final.pth
│   ├── densenet121_plantvillage_final.pth
│   ├── inceptionv3_plantvillage.pth
│   ├── mobilenetv3_plantvillage.pth
│   ├── vgg16_plantvillage.pth
│   └── convnext_tiny_plantvillage.pth
│
├── history/                                     # Training history (.npy)
│   ├── cnn_history.npy
│   ├── swin_history.npy
│   ├── resnet_history1.npy
│   ├── densenet_history.npy
│   ├── inception_history.npy
│   ├── mobilenet_history.npy
│   ├── vgg_history.npy
│   └── convnext_history.npy
│
├── requirements.txt
└── README.md
```

> ⚠️ `.pth` model files are **not included** in this repo due to GitHub's 100MB file limit. Re-train using the notebook or download from Kaggle outputs.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:39ff9d,100:0d1614&height=3&width=100%"/>

## 🚀 How to Run

<details open>
<summary><b>🅰️ Option 1 — Run on Kaggle (Recommended · Free GPU)</b></summary>

1. Go to [Kaggle.com](https://www.kaggle.com) and create an account
2. Click **"+ New Notebook"**
3. Add the PlantVillage dataset: click **"Add Data"** → search `PlantVillage`
4. Upload and run `plant-diseases-classification-system.ipynb`
5. Enable **GPU** under Settings → Accelerator → GPU T4 x2

</details>

<details>
<summary><b>🅱️ Option 2 — Run Locally</b></summary>

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/plant-disease-classifier.git
cd plant-disease-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the PlantVillage dataset from Kaggle
# Place it at: /data/plantvillage/color/

# 4. Open the notebook
jupyter notebook plant-diseases-classification-system.ipynb
```

</details>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:39ff9d,100:0d1614&height=3&width=100%"/>

## 🛠️ Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy
matplotlib
Pillow
scikit-learn
seaborn
pandas
ipywidgets
gradio
grad-cam
```

Install all at once:
```bash
pip install -r requirements.txt
```

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:39ff9d,100:0d1614&height=3&width=100%"/>

## 🎮 Interactive Demo

This project includes **two interactive demo UIs**:

<table>
<tr>
<td width="50%" valign="top">

### 🔹 ipywidgets Dashboard
*(Kaggle Notebook)*
- Upload any leaf image directly inside the notebook
- Get instant prediction + confidence score
- Works inline in Kaggle / Jupyter

</td>
<td width="50%" valign="top">

### 🔹 Gradio Multi-Engine Dashboard
```python
demo.launch(share=True)  # public link!
```
- Choose from **all 8 trained engines** via dropdown
- Enterprise-grade dark-themed diagnostic cards
- 🟢 Healthy · 🔴 Disease Detected
- Confidence bar + recommended action

</td>
</tr>
</table>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:39ff9d,100:0d1614&height=3&width=100%"/>

## 🗺️ Notebook Walkthrough

<details>
<summary><b>Click to expand the full step-by-step notebook flow</b></summary>

| Section | What Happens |
|---------|-------------|
| **Setup & GPU Check** | Detects CUDA, auto-finds dataset path |
| **Noise Visualization** | Side-by-side: clean vs Gaussian-noisy leaf |
| **Data Pipeline** | ImageFolder → random 80/20 split → DataLoaders |
| **EfficientNet-B0 Training** | CNN baseline, 10 epochs, saves `.pth` + `.npy` |
| **Swin Transformer Training** | Full unfreeze, lr=5e-5, 10 epochs |
| **ResNet-50 Training** | Frozen backbone + Dropout head |
| **DenseNet-121 Training** | Feature reuse architecture |
| **Inception-V3 Training** | Special 299×299 loader, aux_logits disabled |
| **MobileNet-V3 Training** | Lightweight model for edge deployment |
| **VGG-16 Training** | Classic deep CNN with multi-GPU support |
| **ConvNeXt-Tiny Training** | AdamW optimizer, modern architecture |
| **Results Table** | Master comparison of all 8 models |
| **Visualization** | Bar charts + learning curves for top models |
| **Noise Stress Test** | CNN vs ViT robustness comparison |
| **Grad-CAM Heatmaps** | Visual explainability for both models |
| **Gradio App** | Production-ready 8-engine diagnostic dashboard |

</details>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:39ff9d,100:0d1614&height=3&width=100%"/>

## 💡 Key Insights

- 🥇 **ConvNeXt-Tiny** achieves the highest accuracy — a modern CNN that borrows design ideas from Vision Transformers
- 👁️ **Swin Transformer** shows superior robustness on noisy images compared to traditional CNNs — global attention mechanisms help generalize better
- 📱 **MobileNet-V3** is the best pick for edge devices (IoT, mobile phones) due to its lightweight size
- 📦 **VGG-16**, despite being older, still performs competitively on well-structured datasets like PlantVillage
- 🔥 **Grad-CAM** confirms that ViT models focus more holistically on the leaf structure, while CNNs sometimes latch onto irrelevant texture patterns

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:39ff9d,100:0d1614&height=3&width=100%"/>

## 🤝 Acknowledgements

<div align="center">

[![PlantVillage](https://img.shields.io/badge/PlantVillage%20Dataset-Abdallah%20Ali-1c8a5c?style=for-the-badge)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning%20Framework-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Grad-CAM](https://img.shields.io/badge/pytorch--grad--cam-Explainability-1c8a5c?style=for-the-badge)](https://github.com/jacobgil/pytorch-grad-cam)
[![Gradio](https://img.shields.io/badge/Gradio-ML%20Demo%20Framework-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Free%20GPU%20Environment-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/)

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:39ff9d,100:0d1614&height=3&width=100%"/>

## 📬 Connect

<div align="center">

If you found this project helpful, feel free to ⭐ **star the repo** and connect!

![Stars](https://img.shields.io/github/stars/YOUR_USERNAME/plant-disease-classifier?style=social)
![Forks](https://img.shields.io/github/forks/YOUR_USERNAME/plant-disease-classifier?style=social)

<br/>

<i>Built with 🌿 and deep learning — for smarter, healthier crops.</i>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1614,50:1c8a5c,100:39ff9d&height=120&section=footer" width="100%"/>

</div>
