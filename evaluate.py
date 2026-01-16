import torch
import glob
from src.datasets import DefectDataset
from src.models import UNet
from src.utils import get_device
from torch.utils.data import DataLoader

device = get_device()

model = UNet().to(device)
model.load_state_dict(torch.load("unet_defect.pt", map_location=device))
model.eval()

images = sorted(glob.glob("data/images/*"))
masks = sorted(glob.glob("data/masks/*"))

loader = DataLoader(DefectDataset(images, masks), batch_size=1)

with torch.no_grad():
    for x, y in loader:
        x = torch.tensor(x).to(device)
        preds = torch.sigmoid(model(x))
        print("Prediction mean:", preds.mean().item())
