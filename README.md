# Industrial-defect-detection
This project focuses on detecting and segmenting subtle defects in industrial inspection imagery, where signal-to-noise ratio is low and defects are visually ambiguous.

A U-Net-based segmentation model is trained on real industrial textures (MVTec AD / NEU dataset), with careful handling of class imbalance, augmentations, and evaluation using IoU/Dice.

The objective is to simulate challenges encountered in real industrial vision pipelines rather than achieving leaderboard performance.


## How to Run
```bash 
pip install -r requirements.txt
python train.py
python evaluate.py
python inference.py

