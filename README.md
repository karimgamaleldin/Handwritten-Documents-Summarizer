# Handwritten Document Summarizer 👀

This projects aim to create an OCR (optical character recognition) by implementing models from scratch and fine-tuning pre-trained vision language models 

## Descriptions 🖊️

This project 

## Tech Stack 💻

<div align="center">

![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=lightning&logoColor=white)
![NumPY](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Javascript](https://img.shields.io/badge/JavaScript-323330?style=for-the-badge&logo=javascript&logoColor=F7DF1E)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Ant Design](https://img.shields.io/badge/Ant%20Design-1890FF?style=for-the-badge&logo=antdesign&logoColor=white)
![Nginx](https://img.shields.io/badge/Nginx-009639?style=for-the-badge&logo=nginx&logoColor=white)
![Docmer](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)

</div>

**Dataset:** [IAM Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)

**Data pre-processing:** NumPy, OpenCV, Pytorch, Albumentations, HuggingFace's Tokenizers

**Model Development:** Pytorch, Pytorch lightning, HuggingFace's Transformers

**Training:** Kaggle

**Client:** React, Ant Design, Axios

**Server:** Flask

**Deployment:** Nginx, Docker

## Installation

Follow these steps to install and set up the project on your local machine using Docker and Docker Compose.

### Prerequisites

Before you begin, ensure you have met the following requirements:
- **Git** - [Download & Install Git](https://git-scm.com/downloads).
- **Docker** - [Download & Install Docker](https://docs.docker.com/get-docker/).
- **Docker Compose** - [official guide](https://docs.docker.com/compose/install/).

### Cloning the Repository

To get started with the project, clone the repository to your local machine:
```bash
git clone https://github.com/karimgamaleldin/Handwritten-Documents-Summarizer.git
cd Handwritten-Documents-Summarizer/web_application
```

### Running the Application

```bash
docker-compose up --build
```

### Stopping the Application

```bash
docker-compose down
```



## Credits

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)
- [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808v1)
- [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282)
- [OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664)
- [Attention is all you need](https://arxiv.org/abs/1706.03762)
