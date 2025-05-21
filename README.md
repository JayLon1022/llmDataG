# LLM Data Generation Tool

This repository contains the code for the paper "LLM_based Immune Detection Method for Unknown Network Attacks in ICS Under Fewshot Conditions".

## Overview

This repository provides a pipeline for cloning and augmenting datasets using Large Language Models (LLMs). Given a prompt template and input vectors, our system leverages the generative power of LLMs to produce augmented, variant vectors, facilitating data enrichment for training robust machine learning models

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/JayLon1022/llmDataG.git
   cd llmDataG
   ```
2. Create and activate a conda environment:

   ```bash
   conda env create -f environment.yml --name llm
   conda activate llm
   ```

## Structure

## Workflow

1. **Prompt Template**: A customizable prompt guides the LLM in processing input vectors.
2. **Input Vector**: Original data provided to the LLM as the starting point.
3. **LLM Processing**: The model generates mutated output vectors based on the provided prompt.
4. **Output Vectors**: Augmented data ready for downstream applications.

## Usage

1. Prepare your input vectors (to be cloned) and use our prompt template or define a new prompt template specifying the desired transformation.
2. Run the provided scripts to automate interaction with the LLM and produce mutated vectors.
3. Output vectors can be directly integrated into your existing datasets.

## Example

* **Prompt Template**: "You are a professional data analyst with expertise in vector analysis and spatial distribution. Task: You will be given a dataset: {input_dataset}. Analyze the dataset's features and its spatial distribution. Based on the provided input vector, generate new vectors that exhibit similar characteristics but fill in gaps in the feature space of the dataset. Each element of the new vectors must be a five-decimal number within the range (0, 1) and must not have zero as the last decimal place. Ensure that each generated number is meaningful and does not contain redundant zeros or repeating digits. Important: Ensure that the generated vectors represent meaningful variations that reflect the underlying spatial structure of the dataset. Provide only the output vectors, without any explanation or process details. Output Format MUST be: 1.23456, 2.34567, 3.45678, ...Note: Focus on the spatial analysis and ensure that new vectors cover areas of the feature space that are underrepresented or not captured by the current dataset.""
* **Input**: `<span>[0.1, 0.5, 0.3, 0.8]</span>`
* **Output**: `<span>[0.12, 0.47, 0.31, 0.79]</span>`
