# Analyzing the Workflows of Routine Second Trimester Fetal Anatomy Ultrasound Scans

## Overview

This repository contains the code and data associated with a thesis submitted in partial fulfillment of the requirements for the degree of Master of Science (MSc) in **Artificial Intelligence for Sustainable Development** at University College London. The focus of this thesis is to analyze the workflows involved in second trimester fetal anatomy ultrasound scans, with the goal of enhancing our understanding of varying scanning proficiencies among sonographers.

## Repository Structure

- `data/`: Contains our raw dataset videos used for training and evaluation (but were not uploaded due to size).
- `final_models/`: Includes all finetuned models used in the thesis.
- `output/`: Contains all of the outputed files, including text detection, frame collection detection, exapnding the new labels, and the predictions of the models.
- `label_conversion_csvs/`: Includes all conversions between ground truth annotated labels and their classes.
- `sononet/`: Taken directly from https://github.com/rdroste/SonoNet_PyTorch/tree/master.
- Other python files included are able to finetune, visualize results, detect labels, and split the dataset into the train and test used.
Note this repository stems from: https://github.com/surgical-vision/proximity-to-sp-us-videos

## Getting Started

1. **Clone the Repository**

   ```bash
   git clone https://github.com/cmktclmc/fetal-ultrasound-workflow-analysis.git](https://github.com/cmktclmc/Analyzing-WFs-of-Fetal-US-Scans/
