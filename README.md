<h1 align="left">EpiCon-Former: A Contrastive Transformer for Generalizable Epigenomic Signal Representation</h1>


## **1. Motivation**

<p align="center">
	<img width="2928" height="1787" alt="readme_motivation" src="https://github.com/user-attachments/assets/b16a5b87-b652-4836-b566-71056e6ef725" />
</p>

Biological datasets often suffer from scarce labels and experimental noise, making robust representation learning essential.
Histone modification data provide abundant and meaningful signals, but require models capable of capturing their complex patterns.
EpiCon-Former tackles this problem using supervised contrastive learning to learn generalizable and biologically grounded epigenomic embeddings.

---

## **2. Overview of the EpiCon-Former architecture**

<p align="center">
	<img width="3084" height="1333" alt="readme_overview" src="https://github.com/user-attachments/assets/ad3158e4-a6d9-4dda-8037-4aeb5f1ed53e" />
</p>


EpiCon-Former is a Transformer-based model for learning universal representations of histone modification signals.
It uses pseudo-label–guided Supervised Contrastive Learning (SupCon) to build a well-structured embedding space through both instance-level and class-level positive pairs.
To improve noise robustness, the framework applies strong augmentations such as signal shifting, dropout, scaling, and Gaussian noise.

The pretrained encoder serves as a flexible backbone that can be fine-tuned for downstream genomic tasks, including compartment prediction, promoter classification, and chromatin state inference.

---

## **3. Objective**

<p align="center">
	<img width="2122" height="623" alt="readme_objective" src="https://github.com/user-attachments/assets/153b6adc-2cee-4205-a37c-a65239dda8b6" />
</p>

This repository provides downstream pipelines built on top of the pretrained EpiCon-Former encoder, with three primary objectives:

- **EpiCon-Comp** — Fine-tune EpiCon-Former to predict A/B chromatin compartments and capture large-scale 3D genome organization.
- **EpiCon-PR** — Adapt the encoder for promoter vs. non-promoter classification to evaluate promoter activity.
- **EpiCon-State** — Predict ChromHMM chromatin states to assess the biological interpretability of learned representations.

---

## **4. Dependencies**

---

## **5. Download Model & Dataset**

---

## **6. License**


