# Wav2Vec2-Based Speech Classification Experiment

This repository contains an experiment using Wav2Vec2 for speech classification, focused on detecting speech disfluency (stuttering) from audio data. The model is trained and evaluated using Hugging Face’s datasets and transformers libraries.

---

## Model
- Architecture: `facebook/wav2vec2-base`
- Task: Audio classification
- Framework: PyTorch with Hugging Face Transformers

---

## Dataset
- Format: Audio samples with corresponding labels
- Audio processing:
  - Resampled to 16 kHz
  - Encoded using `Wav2Vec2Processor`
- Label used: `stutter_label`

---

The following dataset was used for training and evaluation:

### LibriStutter (4.7k)
- **Source:** `stillerman/libristutter-4.7k` (Hugging Face)
- **Sampling Rate:** 16 kHz
- **Task:** Binary speech disfluency classification
- **Label Creation:**  
  A binary label `stutter_label` was generated based on the presence of the `[STUTTER]` token in transcripts.

## Preprocessing
- Batched audio arrays extracted from the dataset
- Padding applied to handle variable-length audio inputs
- Labels attached during preprocessing
- Dataset encoded using `map()` with batching enabled

---

## Training
- Batch size: 8
- Early stopping: Enabled
- Hardware support: GPU (if available)

Early stopping was triggered after no improvement over multiple evaluation steps.

---

## Evaluation Metrics
The following metrics were used to evaluate model performance:
- Accuracy
- F1 score
- Loss

---

## Results

### Validation Results
- Accuracy: 80.79%
- F1 score: 0.761
- Loss: 0.401

### Test Set Results
- Accuracy: 75.95%
- F1 score: 0.687
- Loss: 0.499

---

## Libraries Used
- torch
- torchaudio
- datasets
- transformers
- evaluate
- accelerate

---

## Usage
Run the provided Python script to preprocess the audio data, train the Wav2Vec2 model, and evaluate performance on validation and test sets.

---

## Purpose
This experiment was conducted as part of an academic study exploring speech classification using self-supervised audio models, with a focus on speech disfluency detection.
