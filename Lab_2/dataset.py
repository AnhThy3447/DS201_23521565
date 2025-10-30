import os
import cv2 as cv
import numpy as np
import idx2numpy
import torch
from torch.utils.data import Dataset

def collate_fn(items: list[dict]) -> dict[torch.Tensor]:
    images = [item["image"] for item in items]
    labels = [item["label"] for item in items]

    images = np.stack(images, axis=0)
    if images.ndim == 3:
        images = np.expand_dims(images, axis=1)
    elif images.ndim == 4:
        images = np.transpose(images, (0, 3, 1, 2))

    images = torch.tensor(images, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    return {"image": images, "label": labels}

class MNISTDataset(Dataset):
    def __init__(self, images_path: str, labels_path: str) -> None:
        images = idx2numpy.convert_from_file(images_path)
        labels = idx2numpy.convert_from_file(labels_path)

        self.data = [
            {
                "image": np.array(image),
                "label": label
            }
            for image, label in zip(images.tolist(), labels.tolist())
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        return self.data[index]


class VinaFood21(Dataset):
    def __init__(self, path: str, label_list=None):
        super().__init__()
        self.label_list = label_list if label_list is not None else {}
        self.data = self.scan_data(path)
        self.idx2label = {v: k for k, v in self.label_list.items()}

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)

    def scan_data(self, path):
        data = []
        label_id = len(self.label_list)

        for folder in sorted(os.listdir(path)):
            folder_path = os.path.join(path, folder)
            if not os.path.isdir(folder_path):
                continue

            if folder not in self.label_list:
                self.label_list[folder] = label_id
                label_id += 1

            for image_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, image_file)
                data.append({
                    "image_path": img_path,
                    "label": folder
                })
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        image = cv.imread(item["image_path"])        
        image = cv.resize(image, (224, 224))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        mean = np.array(self.mean)
        std = np.array(self.std)
        image = (image - mean) / std  # 🔹 vẫn shape (224, 224, 3)
        
        label_id = self.label_list[item["label"]]
        return {"image": image, "label": label_id}


