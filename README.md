# Twi Speech Recognition Model
This repository contains the implementation of a speech recognition model for the Twi language. The model was trained, evaluated, and deployed to provide an API endpoint for transcribing Twi audio files. Below is a detailed overview of the project, including the steps taken, the evaluation process, and how to use the deployed model.

# Table of Contents

1. Project Overview

2. Dataset

3. Model Training

4. Evaluation

5. Deployment

6. API Usage

7. Submission

8. Report

9. License

# Project Overview
This project aims to train a speech recognition model for the Twi language. The model was trained using a dataset of Twi audio recordings and their corresponding transcriptions. The final model was evaluated on a held-out test set and a newly compiled test set of at least 20 sentences. The model was then deployed to Hugging Face Spaces, providing an API endpoint for transcribing Twi audio files.

# Dataset
The dataset used for training and evaluation consists of Twi audio recordings paired with their transcriptions. The dataset was split into:

Training set: Used to train the model.

Validation set: Used for hyperparameter tuning and model selection.

Test set: Held out for final evaluation.

Additionally, a new test set of 20 sentences was compiled to further evaluate the model's performance.

# Model Training
## Approach
Pre-trained Models: Leveraged a pre-trained speech recognition model (e.g., Wav2Vec 2.0) and fine-tuned it on the Twi dataset.

Fine-tuning: The model was fine-tuned using the Hugging Face Transformers library.

Training Pipeline:

Preprocess the audio data (resampling, normalization, etc.).

Tokenize the transcriptions.

Fine-tune the pre-trained model on the Twi dataset.

Save the best model checkpoint based on validation performance.

## Tools and Libraries
1. Hugging Face Transformers

2. PyTorch

3. Datasets library for data preprocessing

# Evaluation
### Metrics
The model was evaluated using the following metrics:

Word Error Rate (WER): Measures the accuracy of the transcriptions.

Character Error Rate (CER): Provides a finer-grained evaluation of transcription accuracy.

### Test Sets
Held-out Test Set: A portion of the original dataset was held out for evaluation.

New Test Set: A new set of 20 sentences was compiled to test the model's generalization ability.

### Results
WER on Held-out Test Set: [8.5094]

CER on Held-out Test Set: [8.8481]


### Deployment
The model was deployed to Hugging Face Spaces, providing an API endpoint for transcribing Twi audio files. The deployment includes:

A user-friendly interface for uploading audio files.

A backend that processes the audio and returns the transcription.

### Deployment Link
Twi Speech Recognition Model on Hugging Face : https://huggingface.co/spaces/calvin9090/Twi_Dataset_Team3

### API Usage
To use the deployed model, follow these steps:

Access the API:

Visit the Hugging Face Space: Twi Speech Recognition Model.

Upload Audio:

Use the interface to upload an audio file in a supported format (e.g., WAV, MP3).

Get Transcription:

The model will process the audio and return the transcription.

# Example Code for API Usage

import requests

### API endpoint
API_URL = "https://huggingface.co/spaces/calvin9090/Twi_Dataset_Team3/api/predict"

### Upload audio file
files = {"file": open("path_to_audio_file.wav", "rb")}
response = requests.post(API_URL, files=files)

### Get transcription
transcription = response.json()["transcription"]
print("Transcription:", transcription)

### Submission
The following items are included in the submission:

### Python Notebook:

A clear and well-documented Jupyter notebook containing all code used for training, prediction, and evaluation.

Run Python Notebook:

An executed version of the notebook with all outputs and results.

# Report PDF:

A 2-3 page report summarizing the work done, including:

A brief description of the project.

Justification for choices made during the project.

Evaluation results and analysis.

Challenges faced and future improvements.

# Report Summary
The report provides a detailed overview of the project, including:

Dataset: Description of the dataset and preprocessing steps.

Model: Details of the pre-trained model and fine-tuning process.

Evaluation: Results on both the held-out test set and the new test set.

Deployment: Description of the deployment process and API usage.

Challenges: Discussion of challenges faced and potential improvements.

# License
This project is licensed under the MIT License. See the LICENSE file for details.
