# Basic Level LLM Using Transformer Models

## **Overview**
This project represents my attempt at creating a basic-level Large Language Model (LLM) inspired by the GPT-2 model (124M parameters). The implementation focuses on leveraging transformer architectures to develop a foundational understanding of LLMs and their working principles.

## **Objective**
The goal is to build a transformer-based model capable of generating coherent and contextually relevant text. While not achieving the full complexity of GPT-2, this project is aimed at exploring the key components and techniques used in modern language models.The project will also try and optimise the model using the weights made public by OpenAI for GPT-2


## **Features**
1. **Transformer Architecture:** 
   - Implements the fundamental transformer layers (self-attention, feed-forward networks, etc.) to power the model.
   
2. **OpenWebText Dataset:**
   - Utilizes the OpenWebText dataset for training to replicate the data distribution GPT-2 was exposed to.

3. **Parameter Scale:**
   - Approximately 124M parameters, aligning with GPT-2’s smallest version.

4. **Language Modeling Objective:**
   - Trains the model to predict the next token in a sequence, leveraging standard techniques like causal masking.

5. **Text Generation:**
   - Supports text generation by sampling or greedy decoding methods.
   - Accepts input prompts to produce contextually relevant outputs.

## **Components**
### 1. **Model Architecture**
- **Embedding Layer:**
  Converts input tokens into dense vectors.
  
- **Transformer Block:**
  - Multi-Head Self-Attention: Captures dependencies across the input sequence.
  - Feed-Forward Network: Applies non-linear transformations for feature extraction.
  - Layer Normalization and Residual Connections: Ensures stable gradients during training.

- **Output Layer:**
  Maps the processed vectors back to the vocabulary space to predict the next token.

### 2. **Dataset Preparation**
- OpenWebText is preprocessed into tokenized sequences.
- Causal masking is applied to ensure that the model only attends to past tokens during training.

### 3. **Training Pipeline**
- Loss Function: Cross-entropy loss for next-token prediction.
- 
- Optimizer: AdamW optimizer with weight decay.
- Scheduler: Implements learning rate warm-up followed by cosine decay.

### 4. **Evaluation**
- Qualitative assessment through text generation.

### 5. **Text Generation**
- Sampling methods include:
  - Greedy decoding.
  - Top-k and Top-p sampling for controlled randomness.
- Outputs are evaluated based on relevance and coherence.

## **Challenges**
- Balancing model complexity with computational resource constraints.
- Fine-tuning the hyperparameters for optimal performance.

## **Future Improvements**
1. Scale up the model parameters and layers to approach higher levels of fluency.
2. Experiment with additional datasets to improve generalization.
3. Incorporate pre-training and fine-tuning stages for specific tasks.

## **Acknowledgments**
This project is inspired by OpenAI's GPT-2 model and aims to provide a foundational understanding of LLM development. Special thanks to the creators of the OpenWebText dataset for their open contributions to language model research.

## **Disclaimer**
This is an experimental project and does not aim to replicate GPT-2's full performance. My focus is on learning and exploration rather than production-grade development.
