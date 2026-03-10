import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from unet_model import UNet
from dataset_loader import SatelliteDataset


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

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss}")


torch.save(model.state_dict(), "models/unet_model.pth")

print("Training completed")