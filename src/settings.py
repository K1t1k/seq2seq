import torch


class Settings:
    DATASET = "russian"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_EPOCHS = 10
    BATCH_SIZE = 64
    SEQ_SIZE = 64
    
    N_LAYER = 2
    HIDDEN_SIZE = 256
    LR = 1e-3