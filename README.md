## **1. Motivation**

<p align="center">
  <img width="628" height="357" alt="image" src="https://github.com/user-attachments/assets/b63a98c0-3d1e-459e-8665-a02b75fc0f49" />
</p>

Recent progress in deep learning has shown strong performance in biological data analysis, despite challenges such as limited labels, high noise, and cell-type variability.
Histone modification signals contain rich regulatory information, but require robust representation learning to capture meaningful biological patterns.
EpiCon-Former addresses these challenges by applying supervised contrastive learning to learn noise-tolerant and generalizable epigenomic embeddings from large-scale histone datasets.

---

## **2. Overview**

<p align="center">
  <img width="672" height="268" src="https://github.com/user-attachments/assets/59827765-34ca-450f-b968-755b28f21923" />
</p>


EpiCon-Former is a Transformer-based framework designed to learn universal representations of histone modification signals.
The model leverages pseudo-label–guided Supervised Contrastive Learning (SupCon) to construct a structured embedding space, forming both instance-level and class-level positive pairs.
To enhance robustness against experimental noise, the framework incorporates multiple augmentation strategies, including signal shifting, dropout, Gaussian noise injection, and scaling.

The pretrained encoder serves as a general-purpose backbone that can be fine-tuned for various genomic prediction tasks such as chromatin compartment prediction, promoter identification, and chromatin state classification.

---

## **3. Objective**

<p align="center">
  <img width="421" height="106" src="https://github.com/user-attachments/assets/a9eb6247-585c-45b3-aa13-e7639818dfff" />
</p>

This repository provides downstream pipelines built on top of the pretrained EpiCon-Former encoder, with three primary objectives:
	•	EpiCon-Comp — Fine-tune EpiCon-Former to predict A/B chromatin compartments and capture large-scale 3D genome organization.
	•	EpiCon-PR — Adapt the encoder for promoter vs. non-promoter classification to evaluate promoter activity.
	•	EpiCon-State — Predict ChromHMM chromatin states to assess the biological interpretability of learned representations.

---

## **4. Dependencies**


## **5. Download Model & Dataset**




> Note: Some datasets may require manual download from ENCODE or user-provided preprocessing steps.
>
