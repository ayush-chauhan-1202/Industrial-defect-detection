import cv2
import torch
import numpy as np
from src.models import UNet
from src.utils import get_device

device = get_device()

model = UNet().to(device)
model.load_state_dict(torch.load("unet_defect.pt", map_location=device))
model.eval()

img = cv2.imread("test.jpg")
img = cv2.resize(img, (256,256))
img = img / 255.0
img = np.transpose(img, (2,0,1))[None].astype("float32")

x = torch.tensor(img).to(device)

with torch.no_grad():
    pred = torch.sigmoid(model(x))[0,0].cpu().numpy()

cv2.imwrite("prediction.png", (pred*255).astype("uint8"))
print("Saved prediction.png")
