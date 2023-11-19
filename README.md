# TCR-ESM: employing protein language embeddings to predict TCR-peptide-MHC binding

## Description

This is the GitHub repository for the project **TCR-ESM: employing protein language embeddings to predict TCR-peptide-MHC
binding** by Shashank Yadav, Dhvani Sandip Vora, Durai Sundar, Jaspreet Kaur Dhanjal

## Getting Started

### Dependencies / Requirements
* numpy==1.23.5
* pandas==1.5.3
* tqdm==4.66.1
* natsort==8.4.0
* matplotlib==3.7.1
* tensorflow==2.14.0
* torch==2.1.1
* fair-esm==2.0.0
* scikit-learn==1.2.2

## Repository directories & files
### Embedding Extraction Note: Using ESM1v to generate embeddings of 100 randomly generated peptides using ESM1v takes approximately 17 seconds, model training time may vary from 3 minutes to 3 hours, and predictions involve a quick forward pass lasting up to 3 seconds. These computations were performed on a GPU with 16 GB VRAM, demonstrating the computational efficiency of TCR-ESM. This highlights the feasibility of implementing TCR-ESM for scanning numerous peptides swiftly, aligning to facilitate cognate target identification for T-cell receptors in the context of T-cell therapy development.
+ [`0_ESM_Embeddings_Extractor.ipynb`](0_ESM_Embeddings_Extractor.ipynb) contains code for extraction of embeddings from the ESM1v model using the fair-esm library.



