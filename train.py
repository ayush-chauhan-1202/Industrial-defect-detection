import glob
import torch
from torch.utils.data import DataLoader
from src.datasets import DefectDataset
from src.models import UNet
from src.losses import DiceLoss
from src.utils import get_device
from tqdm import tqdm

device = get_device()
print("Using:", device)

images = sorted(glob.glob("data/images/*"))
masks = sorted(glob.glob("data/masks/*"))

dataset = DefectDataset(images, masks)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = DiceLoss()

for epoch in range(10):
    model.train()
    total_loss = 0

    for x, y in tqdm(loader):
        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)

        preds = model(x)
        loss = criterion(preds, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: loss = {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "unet_defect.pt")
