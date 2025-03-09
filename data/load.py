import json
from pathlib import Path
import os
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
from deepface import DeepFace


class FaceDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        face_img = cv2.imread(img_path)
        if face_img is None:
            raise ValueError(f"Failed to read image: {img_path}")

        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_img)

        if self.transform:
            face_img = self.transform(face_img)

        if self.labels is not None:
            label = self.labels[idx]
            return face_img, label
        else:
            return face_img, self.image_paths[idx]


def init_dataloader(image_paths, labels, transform, batch_size=32, shuffle=True):
    dataset = FaceDataset(image_paths, labels, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def get_deepface_pred(image_path):
    analysis = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False)
    emotion = analysis[0]['dominant_emotion']
    prob = analysis[0]['emotion']['happy']
    if emotion == "happy":
        return 1, prob
    else:
        return 0, prob


def create_deepface_pred(image_paths):
    non_labelled_image_deepface = {}
    from tqdm import tqdm
    for path in tqdm(image_paths):
        label, prod = get_deepface_pred(path)
        non_labelled_image_deepface[path] = [label, prod]
    with open("/dataset/deepface_label.json", "w", encoding="utf-8") as json_file:
        json.dump(non_labelled_image_deepface, json_file, ensure_ascii=False, indent=4)
    return non_labelled_image_deepface


def get_data():
    current_path = Path().resolve()
    print("Current Path:", current_path)
    base_dir = current_path / "dataset" / "lfw_funneled"
    all_folders = [os.path.join(base_dir, person) for person in os.listdir(base_dir) if
                   os.path.isdir(os.path.join(base_dir, person))]
    train_folders = list(sorted(all_folders))[:-75]
    test_folders = list(sorted(all_folders))[-75:]
    label = open(current_path / "dataset" / "label.txt", "r").read().splitlines()
    label = list(map(int, label))
    non_labelled_image = []
    labelled_image = []
    total_images = 0
    for path in train_folders:
        images = [os.path.join(path, img) for img in os.listdir(path)]
        if images:
            if total_images < len(label):
                labelled_image += images
            else:
                non_labelled_image += images
    return labelled_image, non_labelled_image, label, test_folders
