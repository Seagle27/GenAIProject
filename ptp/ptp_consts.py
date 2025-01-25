import torch

# CONSTANTS:
MAX_NUM_WORDS = 77
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
LOW_RESOURCE = False
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')