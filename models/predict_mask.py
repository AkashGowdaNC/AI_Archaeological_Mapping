import cv2
import torch
import numpy as np

from unet_model import UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = UNet().to(device)
model.load_state_dict(torch.load("models/unet_model.pth"))
model.eval()


image_path = "sample_satellite.jpg"

image = cv2.imread(image_path)
image = cv2.resize(image, (256,256))
image = image / 255.0

image = np.transpose(image, (2,0,1))
image = np.expand_dims(image, axis=0)

image = torch.tensor(image, dtype=torch.float32).to(device)


with torch.no_grad():

    prediction = model(image)

    prediction = torch.argmax(prediction, dim=1)

    mask = prediction.cpu().numpy()[0]


cv2.imwrite("outputs/predicted_mask.png", mask)

print("Prediction saved")
