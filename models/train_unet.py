import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# ===============================
# DATASET CLASS
# ===============================

class SatelliteDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(
            self.mask_dir,
            self.images[index].replace(".jpg", ".png")
        )

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        image = image / 255.0

        image = np.transpose(image, (2, 0, 1))

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


# ===============================
# SIMPLE UNET MODEL
# ===============================

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        self.down1 = DoubleConv(3, 64)
        self.down2 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.up1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, 4, 1)

    def forward(self, x):

        d1 = self.down1(x)
        p1 = self.pool(d1)

        d2 = self.down2(p1)

        up = nn.functional.interpolate(d2, scale_factor=2)

        up = self.up1(up)

        out = self.final(up)

        return out


# ===============================
# TRAINING PIPELINE
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = SatelliteDataset(
    "dataset/train/images",
    "dataset/train/masks"
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = UNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10

for epoch in range(EPOCHS):

    total_loss = 0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")


# ===============================
# SAVE MODEL
# ===============================

torch.save(model.state_dict(), "models/unet_model.pth")

print("Model training completed and saved.")