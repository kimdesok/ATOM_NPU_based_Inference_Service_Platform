# Inference Service Platform for Cancer Diagnosis(Prototype)

## Project Overview

This page describes on-going experiments and their outcomes along with implementation of components to build an inference service platform for cancer diagnosis based on microscopic tissue images.   Currently the following tasks are planned and some of them have been implemented:
* design and implementation of a RESTful client–server inference architecture
* benchmarking of the performance of deep learning models
* validation of the feasibility of a NPU-based production deployment
* requirement analysis of `a pre-production research platform` to explore the architecture and operational workflow of a real clinical AI inference service.

🎯 **Specific Aims:**

1. Design and implement a REST API–based client–server web application as a production-oriented inference platform prototype
2. Benchmark GPU vs. NPU inference performance of CNN-MIL and ViT-MIL models
3. Support batch model compilation and deployment
4. Validate pathology AI service workflow (future work)

### System Architecture <br>
Client (Web / Streamlit) <br>
<div align="left">           │ </div>
<div align="left">           │    REST API   </div>
<div align="left">           ▼</div> 
<div align="left"> Inference Server (FastAPI)</div>

 ├── Model Loader<br>
 ├── Batch Compiler<br>
 ├── Runtime Selector (GPU / NPU)<br>
 └── Performance Profiler<br>

## Current Progress
### Key Features

* Comparison between client vs. server inference execution as a part of validation and verification (no data but they were exactly same)
* One click batch compilation of GPU models for NPU deployment

<img width="607" height="400" alt="image" src="https://github.com/user-attachments/assets/3b67d9ce-6ce6-4537-b3a6-3244891e3982" /><br>
Figure 1 클라이언트 앱 기능 시연: GPU(.keras 포맷)로 학습된 모델을 NPU(.rbln) 포맷으로 배치 컴파일하는 기능 시연<br>

* Hardware-aware runtime selection
* Integrated performance measurement

<img width="643" height="350" alt="image" src="https://github.com/user-attachments/assets/b09485ca-a258-4a7c-b503-7d92e1b71008" /><br>
Figure 2 클라이언트 앱 기능 시연: 실험 모델의 영상분류 성능을 정확도와 산출속도(Throughput)로 표시하고 각 실험 결과는 로깅함(Run history패널)<br>

### Hardware Environment
* GPUs - NVIDIA T4, NVIDIA A100, NVIDIA H100
(additional GPUs depending on environment)
* NPU - Rebellions ATOM PLUS

## Results
### MIL Model Development for WSI-Based Diagnosis

**Hugging Face Space app** available:<br>
<a href="https://huggingface.co/spaces/kimdesok/camelyon17-hitl-dashboard#in-the-planning-stage">
  <img src="images/HF-Space1.png" alt="Alt Text" width="300">
</a>

* Migration from Tensorflow/Keras Framework to `Pytorch ecosystem` that will enable the NPU compatible compilation of vsion foundation models provided by Hugging Face
* Preprocessed datasets need to be further processed or converted to Pytorch equivalents
(ex. from TFRecord to `IterableDataset` by tfrecord-dataset or `WebDataset/LMDB` to store and stream large-scale datasets)
* Internal A100 GPU support was discontinued and some model training was performed to Elice Group-provided A100 compute resources.
* The new environment enabled:
>- large-scale dataset acquisition
>- preprocessing
>- transformer-based MIL model development

### Dataset Acquisition & Processing
**Secured Datasets**
* TCGA-BRCA (WSI, breast cancer histopathology, converted into TFRecord format for high-throughput training
* PatchCamelyon (patches curated from breast cancer lymph node metastasis WSIs available as Hugging Face dataset)
* CAMELYON16 (WSI, breast cancer lymph node metastasis, converted into TFRecord format)
* CAMELYON17 (Planned, scheduled for acquisition in the second half of the year)<br>

**Training status**
- TCGA-BRCA	: MIL training completed
- PatchCamelyon	: completed
- CAMELYON16	: completed
- CAMELYON17	: Planned acquisition but completed with CAMELYON17-WILDS dataset (curated patches)

### Storage Constraint
* Current storage: 2 TB
* Limitations: Cannot store more than two datasets simultaneously <br>
* Each dataset requires: ~5 days for download and preprocessing  <br>

➡️ Required for full pipeline operation: ≥ 5 TB storage

This is a critical requirement for:
* multi-dataset training
* cross-domain generalization experiments

### Model Development
1️⃣ ViT-Based MIL (ViT-MIL)
* Vision Transformer backbone
* Designed for WSI bag-level classification

2️⃣ CNN-Based MIL
* For performance comparison: ResNet50, VGG19, Inception V3
* All CNN backbones were successfully trained in MIL configuration.

📊 Performance <br>
* TCGA-BRCA Test Set 
>- CNN	— (OOM – 160GB memory limit) 
>- CNN-MIL	≥ 0.934 
>- ViT-MIL	≥ 0.970 
>- CNN tile-based training failed due to GPU memory limitations (OOM at 160 GB), while MIL-based approaches enabled scalable WSI learning.

Table 1. Performance comparison between CNN-MIL vs. ViT-MIL for TCGA-BRCA dataset<br>
<img width="583" height="101" alt="image" src="https://github.com/user-attachments/assets/7097d5dd-8da6-437d-86d2-4d6af35646c8" />

* Camelyon16 Test Set <br>
>- CNN-MIL	≥ 0.937 
>- ViT-MIL	≥ 0.989 

Table 2. Performance comparison between CNN-MIL vs. ViT-MIL for Camelyon16 dataset<br>
<img width="545" height="51" alt="Image" src="https://github.com/user-attachments/assets/b1810801-2b70-4c9e-80ef-70ac834d3a6a" />

🔍 **Key Technical Insight**
* ViT-MIL shows clear performance advantage for WSI classification(See Table 1 above).
* MIL is a scalability enabler, not just a modeling choice.
* Storage capacity is a core infrastructure requirement, not an operational detail.
* Visualization of individual patch with its inference outcome show why the model behaves poorly on some patches.

🧪 **Exploration & Planning**<br>
- **To visualize a big picture of and enable an interaction with WSIs**<br>
<img width="321" height="412" alt="image" src="https://github.com/user-attachments/assets/09c12d3f-15fe-408b-8b15-b9e47cb72678" />
Fig. 3 Visualization of pixel level cell type annotations as a part of EDA experiment (Marking of metastatic cancer in orange, tissue in green, and glass black ground in red) Each square represents a 224x224 patch image <br><br>

- **To introduce a morphology based feature embedding into the model training**<br>
<img width="600" height="650" alt="Image" src="https://github.com/user-attachments/assets/7a44b5f5-f0e1-44a8-aa98-e80f028d1ff2" />
Fig. 4 Heatmap for a false negative outcome shows a large background included in the patch(Upper panel:Before tissue masking). When the background is set to 255, the inference outcome changes but still remained as false negative(Lower panel:After tissue masking).  It might be caused by the background that gets attention due to the current design of self attention scheme in the ViT architecture. <br><br>

<img width="643" height="307" alt="Image" src="https://github.com/user-attachments/assets/3e0601d2-3d03-4544-a55f-0e02e975ff8d" />
Fig. 5 A new multihead self attention scheme for ViT backbone has been designed to include nuclear segmentation embedding to represent a morphology based information to force the attention to be given to the high resolution features of the cell nuclei.







