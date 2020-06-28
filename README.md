# KDD2020-Multimodalities-Recall

## Team: Suicide Squad

## Introduction
- An end-to-end and lightweight model, using TensorFlow Estimator API.
- Negative Sampling on the fly
- Ranked 4th among solo teams
- Ranked 25/1433 in final phase

## Pre-processing
- Image Metadata Feature  
  - We generate relative size and relative position, together with class labels, as image metadata features

- Image Feature
  - We directely use the provided feature as image feature.

- Text Feature
  - We process the text query with lemmatization and tokenization.

## Model Architecture
- Text Embedding
  - We use pre-trained normalized word2vec embedding layer and transformer layer to extract intermediate text embedding.

- Image Embedding
  - We firstly generate image metadata input embedding and image input embedding with DNN.
  - The image metadata input embedding should contain position and class information.
  - We fuse the image input embedding and image metadata input embedding together and process with transformer layers to capture the image information and relative position information all together, similar to DETR[1].

- Negative Sampling
  - We conduct in-batch negative sampling on (text embedding, image embedding) pairs. Note that negative sampling should be conducted before cross attention to avoid leakage.

- Mathching
  - We use cross-attention and gated-fusion to better takes comprehensive and fine-grained cross-modal interactions into account, similar to CAMP[2].

- Training
  - We label positive image and text pairs as 1, and negative pairs as 0, and use binary logloss as the loss function for training.

## Reflection
- Pre-trained Text Embedding
  - We should have used BERT, but doesn't have enough computing resource to finetune.

- Negative Sampling
  - We use in-batch negative sampling for efficiency, but this sampling method doesn't upsample hard negastive example which is not ideal in this competition setting.
  - Top1 team used a negative sampling method based on TF-IDF to sample negative query. This is highly effective in practice but ideally we want to directly sample negative query in embedding space. Adversarial negative sampling or something similar to WARP loss would be ideal.
  - Top8 Team used the last word's embedding to find negative query. The logic is that the last word is usually a noun which indicates the type of product. This is more elegant though the embedding space for finding negative query is pre-trained word2vec which is not the final model's text embedding space.

- Loss Function
  - Focal Loss should be used for (at least partially) preventing the easy negative example problem.

- Model Architecture
  - Our model architecture is not overparameterized enough. We are on right direction in terms of architecture (transformer and cross-attention), but should have used more Transformer or BERT layers to extract information and more cross-attention layers to co-attend.
  - MCAN[3] and VisualBERT[4] can be considered.



## Reference
[1] End-to-End Object Detection with Transformers  
[2] CAMP: Cross-Modal Adaptive Message Passing for Text-Image Retrieval  
[3] Deep Modular Co-Attention Networks for Visual Question Answering  
[4] VisualBERT: A Simple and Performant Baseline for Vision and Language  
