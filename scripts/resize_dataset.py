import os
import cv2

image_folder = "dataset/images"
mask_folder = "dataset/masks"

for file in os.listdir(image_folder):

    img = cv2.imread(os.path.join(image_folder,file))
    mask = cv2.imread(os.path.join(mask_folder,file.replace(".jpg",".png")))

    img = cv2.resize(img,(256,256))
    mask = cv2.resize(mask,(256,256))

    cv2.imwrite(os.path.join(image_folder,file),img)
    cv2.imwrite(os.path.join(mask_folder,file.replace(".jpg",".png")),mask)

print("Resizing completed")