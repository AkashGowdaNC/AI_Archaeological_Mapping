import os

image_folder = "dataset/images"
mask_folder = "dataset/masks"

images = os.listdir(image_folder)
masks = os.listdir(mask_folder)

image_ids = set([i.split(".")[0] for i in images])
mask_ids = set([m.split(".")[0] for m in masks])

missing_masks = image_ids - mask_ids
missing_images = mask_ids - image_ids

print("Total Images:", len(images))
print("Total Masks:", len(masks))

print("Images without masks:", missing_masks)
print("Masks without images:", missing_images)