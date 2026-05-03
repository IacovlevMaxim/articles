# From CLIP to LLaVE: Smarter Training for Multimodal Embeddings

## Article Summary

Large Multimodal Models (LMMs) have become an industry-standard for daily AI users. For example, Claude and Gemini models are able to work with text, image, voice etc. simultaneously for a larger scope of usability. 

Multimodal models are based on a fundamental mechanism called **embedding**. Embedding is a the process of converting data into vectors, where each dimension of the vector encodes some characteristic of the data. These embeddings share the vector space between different data modalities. For example, a vector for the text "scary clown" will be very similar to the image of Pennywise. 

This multimodal approach allows for various applications, such as text-to-image or image-to-text search, document analysis, medical diagnostics and robotics control. However, the performance of LMMs is significantly based on the quality of the embedding. 

**LLaVE** (Large Language and Vision Embedding Model) became an innovation in training multimodal embeddings introduced by Tencent. This article explains how those embeddings are learned, what was lacking in previous approaches, and how LLaVE brought a state-of-the-art performance to multimodal embedding. 

## Get Started

Use the following Google Colab link to get started with the codebase: [link]