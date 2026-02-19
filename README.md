# Inference Service Platform (Prototype) for Supporting Cancer Diagnosis 

GPU ↔ NPU performance benchmarking and RESTful inference service for pathology AI

## Project Overview

This project presents a prototype platform for pathology image–based cancer diagnosis support inference services, designed to:
* benchmark deep learning model inference performance across GPU and NPU
* implement a RESTful client–server inference architecture
* validate NPU-based production deployment feasibility

The system was developed as a pre-production research platform to explore the architecture and operational workflow of a real clinical AI inference service.

## Objectives

* Design and implement a production-oriented inference platform prototype
* Compare GPU vs. NPU inference performance
* Develop a REST API–based client–server web application
* Support batch model compilation and deployment
* Validate pathology AI service workflow

### System Architecture <br>
Client (Web / Streamlit) <br>
                │<br>
                │ REST API<br>
                ▼<br>
Inference Server (FastAPI)<br>
 ├── Model Loader<br>
 ├── Batch Compiler<br>
 ├── Runtime Selector (GPU / NPU)<br>
 └── Performance Profiler<br>

### Key Features

* Client–server inference execution mode comparison
* Batch model compilation for NPU deployment
* Hardware-aware runtime selection
* Integrated performance measurement

### Hardware Environment
* GPUs - NVIDIA T4, NVIDIA A100, NVIDIA H100
(additional GPU depending on environment)

* NPU - Rebellions ATOM+

## Results
### MIL Model Development for WSI-Based Diagnosis
Infrastructure Migration
Internal A100 GPU support was discontinued.
Training was migrated to Elice-provided A100 compute resources.
The new environment enabled:
large-scale dataset acquisition
preprocessing
transformer-based MIL model development

### Dataset Acquisition & Processing
All training datasets were converted into TFRecord format for high-throughput training.
Secured Datasets
*TCGA-BRCA (breast cancer histopathology)

PatchCamelyon (lymph node metastasis classification)

*CAMELYON16 (WSI metastasis detection)

*Planned - CAMELYON17 (scheduled for acquisition in the second half of the year)

### Storage Constraint

*Current storage: 2 TB
Limitations: Cannot store more than two datasets simultaneously
Each dataset requires: ~5 days for download and preprocessing

➡️ Required for full pipeline operation: ≥ 5 TB storage

This is a critical requirement for:
*multi-dataset training
*cross-domain generalization experiments

### Model Development
1️⃣ ViT-Based MIL (ViT-MIL)
Vision Transformer backbone
Designed for WSI bag-level classification

2️⃣ CNN-Based MIL
For performance comparison: ResNet50, VGG19, Inception V3

All CNN backbones were successfully trained in MIL configuration.

🧪 Training Status
Dataset	Status
TCGA-BRCA	- MIL training completed
CAMELYON16	-Dataset analysis in progress
PatchCamelyon	- Scheduled for MIL training
CAMELYON17	- Planned acquisition
📊 Performance (TCGA-BRCA Test Set)
Model Type	AUC
CNN	— (OOM – 160GB memory limit)
CNN-MIL	≥ 0.90
ViT-MIL	≥ 0.95

CNN tile-based training failed due to GPU memory limitations (OOM at 160 GB),
while MIL-based approaches enabled scalable WSI learning.

🔍 Key Technical Insight
*MIL is not just a modeling choice — it is a scalability enabler
*ViT-MIL shows clear performance advantage for WSI classification
*Storage capacity is a core infrastructure requirement, not an operational detail
 
표 1. CNN을 backbone으로 한 MIL 모델과 ViT를 backbone으로 한 ViT-MIL과의 정확도 비교<br>
<img width="583" height="101" alt="image" src="https://github.com/user-attachments/assets/7097d5dd-8da6-437d-86d2-4d6af35646c8" />

<img width="536" height="280" alt="image" src="https://github.com/user-attachments/assets/e570cca0-d245-46c0-ba1b-d2605e84d81b" /><br>
그림 1.  EDA 실험의 일부로서 픽셀 수준의 어노테이션 정보(녹색실선)의 영역(region)별 가시화<br>

<img width="321" height="412" alt="image" src="https://github.com/user-attachments/assets/09c12d3f-15fe-408b-8b15-b9e47cb72678" /><br>
그림 2 EDA 실험의 일부로서 전체슬라이드영상에서 조직 영역(녹색 실선)과 유리 슬라이드 백그라운드 영역(적색 실선)으로 분할하고 전이암 조직 어노테이션 정보(오렌지색 실선)를 가시화. 사각형 타일은 224x224 패치영상을 나타냄.<br>

<img width="546" height="470" alt="image" src="https://github.com/user-attachments/assets/cd1730f7-7b5d-401c-a3f1-84a0bbc14367" /><br>
그림 3 좌측 슬라이드(A)에서는 암조직 영역이 약 1천여개의 패치 영상으로 이루어져 있지만, 우측 슬라이드(B)에서는 그 영역이 단지 2개의 패치 영상으로 이루어져 있음. EDA 실험의 일부로서 현재 학습 중인 ViT-MIL 모델 학습의 어려운 점과 추론 모델을 활용해서 B 슬라이드 영상을 암으로 분류하기는 매우 어려울 것이라고 예상할 수 있음.  그러므로, 전체슬라이드영상 분류 모델의 차선책으로 패치영상 분류 모델, Semantic Segmentation 모델 등 학습 실험이 수행되어야 함.<br>

<img width="423" height="253" alt="image" src="https://github.com/user-attachments/assets/966b09e5-cf38-4277-8af6-a27673938bfe" /><br>
그림 4 EDA 실험의 일부로서 일정 크기(224x224)의 패치영상을 가시화. 전이암조직 영상이 캡쳐된 패치(적색 하일라이트)와 정상 림프조직 세포 패치영상의 형태학적 비교를 통해 패치수준의 영상분류 모델 학습이 가능하리라는 잠정적 결론의 근거 제공.<br>

<img width="607" height="400" alt="image" src="https://github.com/user-attachments/assets/3b67d9ce-6ce6-4537-b3a6-3244891e3982" /><br>
그림 5 클라이언트 앱 기능 시연: GPU(.keras 포맷)로 학습된 모델을 NPU(.rbln) 포맷으로 배치 컴파일하는 기능 시연<br>

<img width="643" height="350" alt="image" src="https://github.com/user-attachments/assets/b09485ca-a258-4a7c-b503-7d92e1b71008" /><br>
그림 6 클라이언트 앱 기능 시연: 실험 모델의 영상분류 성능을 정확도와 산출속도(Throughput)로 표시하고 각 실험 결과는 로깅함(Run history패널)<br>

