import os, torch

# Funkcija modeļa saglabāšanai
def save_ckpt(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

# Standarta Cityscapes klašu krāsu shēma
CITYSCAPES_TRAINID_COLORS = [
    (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153),
    (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
    ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
    (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32)
]

# Iekārtas veida noteikšanas funkcija
def get_device():
    if torch.cuda.is_available():   # Vai CUDA (nVidia GPU) ir pieejams
        return torch.device("cuda")
    # Vai Apple Silicon MPS ir pieejams. NB! MPS ir pieejams uz M-serijas procesoriem no macOS 12.3+, PyTorch 1.12+
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    # Citādi - vienkārši uz procesora
    return torch.device("cpu")
