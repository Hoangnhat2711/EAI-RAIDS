# config.py

import torch

class Config:
    # General settings
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_STATE = 42
    NUM_CLASSES_ADULT = 2 # Binary classification for income
    NUM_CLASSES_COMPAS = 2 # Binary classification for recidivism

    # Training settings
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS_PER_TASK = 5

    # EWC (Elastic Weight Consolidation) settings
    LAMBDA_REG = 0.1 # Regularization strength for EWC

    # LwF (Learning without Forgetting) settings
    ALPHA_LWF = 1.0 # Distillation loss strength for LwF
    TEMPERATURE_LWF = 2.0 # Temperature for softmax distillation in LwF

    # Data settings
    ADULT_TEST_SIZE = 0.2
    COMPAS_TEST_SIZE = 0.2

    # FairFace (Image Data) settings
    NUM_CLASSES_FAIRFACE = 7 # Example: 7 races/ethnicities
    FAIRFACE_IMAGE_SIZE = (64, 64) # Height, Width
    FAIRFACE_NUM_CHANNELS = 3 # RGB images
    FAIRFACE_NUM_SENSITIVE_GROUPS = 7 # Example: corresponding to races
    FAIRFACE_NUM_SAMPLES = 1000 # Number of dummy samples to generate
    FAIRFACE_TEST_SIZE = 0.2

    # MLflow settings (if needed, can be managed outside config too)
    # TRACKING_URI = "http://localhost:5000" # Example MLflow tracking URI
