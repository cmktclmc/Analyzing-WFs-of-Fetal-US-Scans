# Analyzing the Workflows of Routine Second Trimester Fetal Anatomy Ultrasound Scans

## Overview

This repository contains the code and data associated with a thesis submitted in partial fulfillment of the requirements for the degree of Master of Science (MSc) in **Artificial Intelligence for Sustainable Development** at University College London. The focus of this thesis is to analyze the workflows involved in second trimester fetal anatomy ultrasound scans, with the goal of enhancing our understanding of varying scanning proficiencies among sonographers.

## Abstract

Fetal ultrasound scanning is a routine procedure used to monitor fetal growth during pregnancy. This procedure involves navigating an ultrasound probe to capture images aligned with specific anatomical planes (e.g., femur, brain, abdomen). The International Society of Ultrasound in Obstetrics and Gynecology (ISUOG) guidelines emphasize the importance of proper alignment but provide minimal standardization on the sequence of image collection. This lack of standardization can result in varying workflows between novice and experienced sonographers.

To address this issue, this project utilizes the pre-trained deep learning model **SonoNet**, which was developed for analyzing fetal ultrasound videos. We test this model on a dataset of second trimester ultrasound videos from both expert and novice sonographers. By fine-tuning SonoNet and comparing its performance with existing studies, we aim to gain insights into the model’s generalizability across different sonographers and identify opportunities to optimize scanning workflows for improved training and patient outcomes.

## Repository Structure

- `data/`: Contains our raw dataset videos used for training and evaluation.
- `src/`: Includes the implementation of the SonoNet model, fine-tuning scripts, and workflow analysis code.
- `notebooks/`: Jupyter notebooks with exploratory data analysis and results visualization.
- `results/`: Output from model evaluations and comparisons with published studies.
- `docs/`: Documentation related to the project, including methodology and guidelines.

## Getting Started

1. **Clone the Repository**

   ```bash
   git clone https://github.com/cmktclmc/fetal-ultrasound-workflow-analysis.git](https://github.com/cmktclmc/Analyzing-WFs-of-Fetal-US-Scans/
