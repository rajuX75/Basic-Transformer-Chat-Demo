# Basic Transformer Chat Demo

This project implements the core architectural components of a Transformer model from scratch in TensorFlow/Keras, demonstrating its application to a simplified conversational chat task. It is structured as a Google Colab notebook for easy execution.

**Disclaimer:** This is a **demonstration project** for learning purposes. It uses a relatively small model and dataset and does not incorporate advanced techniques found in large, production-level language models (like vast pre-training, Reinforcement Learning from Human Feedback (RLHF), Constitutional AI, sophisticated prompting, complex memory, or robust safety layers). Performance will be limited compared to models like ChatGPT.

## Project Overview

The goal of this project is to build and train a basic Encoder-Decoder Transformer model to generate responses in a conversational setting. The implementation highlights fundamental building blocks of the Transformer architecture as described in the "Attention Is All You Need" paper.

The project is implemented as a multi-cell Google Colab notebook, guiding you through the process from data preparation to inference.

## Key Components Implemented

Based on the project goals, the following core components are built:

1.  **Core Transformer Architecture:**
    *   **Self-Attention Mechanisms:** Scaled Dot-Product Attention and Multi-Head Attention.
    *   **Point-wise Feed-Forward Networks:** Simple dense layers applied independently to each position.
    *   **Layer Normalization:** Applied after residual connections.
    *   **Residual Connections:** Adding input to the output of sub-layers to help with gradient flow.
    *   **Positional Encoding:** Injecting information about token position into embeddings.
    *   **Masking:** Padding masks and look-ahead masks for attention.
2.  **Basic Seq2Seq Implementation:** Structure of an Encoder and a Decoder stacking the core layers.
3.  **Data Preparation:** Downloading, parsing, cleaning, tokenizing, and padding text data.
4.  **Basic Training Loop:** Using a custom learning rate schedule, Adam optimizer, and masked cross-entropy loss.
5.  **Greedy Inference:** Generating responses token by token.

## Dataset

The project uses the **Cornell Movie-Dialogs Corpus**. This dataset contains a large number of conversational exchanges extracted from movie scripts.

*   The dataset is downloaded and processed automatically within the Colab notebook.
*   Conversations are broken down into input-target pairs (previous turn, next turn).

## Getting Started

1.  **Open the Colab Notebook:** Access the main project code by opening the `chat_transformer_colab.ipynb` file in Google Colab.
2.  **Set up Runtime:** In Colab, go to `Runtime -> Change runtime type` and select `GPU` as the hardware accelerator.
3.  **Run Cells Sequentially:** Execute each code cell in the notebook from top to bottom.
    *   **Cell 1:** Imports necessary libraries and checks GPU availability.
    *   **Cell 2:** Downloads, extracts, parses, cleans, tokenizes, and prepares the Cornell Movie-Dialogs Corpus. This will define `tokenizer`, `MAX_LENGTH`, `VOCAB_SIZE`, `BATCH_SIZE`, and `dataset`.
    *   **Cell 3:** Defines model and training hyperparameters based on values from Cell 2.
    *   **Cell 4-7:** Defines the core Transformer architectural building blocks (Positional Encoding, Masks, Attention, FFN, Encoder/Decoder Layers).
    *   **Cell 8:** Defines the full Encoder, Decoder, and Transformer models using the layers.
    *   **Cell 9:** Sets up the optimizer, loss function, and metrics.
    *   **Cell 10:** Defines the `@tf.function` for a single training step and sets up checkpointing.
    *   **Cell 11:** Runs the main training loop over the dataset for a specified number of epochs. This will train the `transformer` model and save checkpoints.
    *   **Cell 12:** Loads the latest checkpoint and provides an interactive loop to generate responses from the trained model.

## Caveats and Limitations

It is important to reiterate the limitations of this project:

*   **Small Scale:** The model size and the dataset size (relative to state-of-the-art models) are small.
*   **Training from Scratch:** Training a general-purpose conversational model from scratch requires vastly more data and computation. The training here is a basic demonstration and will likely result in limited generalization.
*   **Basic Inference:** Uses simple greedy decoding. More advanced decoding strategies (beam search, sampling) can yield better results.
*   **No Advanced Techniques:** Lacks crucial components of production systems:
    *   **Massive Pre-training:** Not pre-trained on a broad text corpus.
    *   **Alignment:** No RLHF, Constitutional AI, or other methods to align outputs with human preferences or safety guidelines.
    *   **Memory/Context:** Handles only single-turn Q&A pairs. Does not maintain conversation history or state.
    *   **Robust Safety/Moderation:** No mechanisms to prevent harmful or biased outputs.
    *   **Efficient Serving:** Not optimized for low-latency, high-throughput inference.

For building a more capable chat system, starting with large pre-trained models (e.g., from Hugging Face Transformers) and fine-tuning them is the recommended approach rather than training from scratch with limited resources.

## Future Work (Potential Extensions)

*   Implement more advanced decoding strategies (Beam Search, Top-K/Top-P Sampling).
*   Expand data handling to support multi-turn conversations.
*   Experiment with larger model hyperparameters if computational resources allow.
*   Integrate techniques for managing conversation history or state.
*   Explore fine-tuning a pre-trained model instead of training from scratch.

## License

[MIT License]

## Acknowledgments

*   Inspired by the "Attention Is All You Need" paper (Vaswani et al., 2017).
*   Uses the Cornell Movie-Dialogs Corpus (Danescu-Niculescu-Mizil & Lee, 2011).
