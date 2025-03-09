from sklearn.metrics import confusion_matrix
from PIL import Image
import torch

def test(model, image_paths, val_transform, device):
    model.eval()
    results = []
    single_input = False

    if not isinstance(image_paths, list):
        image_paths = [image_paths]
        single_input = True

    with torch.no_grad():
        for data in image_paths:
            if isinstance(data, str):
                img = Image.open(data).convert("RGB")
            elif isinstance(data, Image.Image):
                img = data.convert("RGB")
            else:
                raise ValueError("Invalid input data type")
            img_tensor = val_transform(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            prob1 = probs[0, 1].item()  # model prediction probability for class 1
            label = 1 if prob1 >= 0.5 else 0  # model prediction label
            results.append((label, prob1 if label == 1 else 1 - prob1))

    return results[0] if single_input else results


def calculate_accu(pred, truth):
    # F1 score
    TP = sum(1 for a, b in zip(pred, truth) if a == 1 and b == 1)
    FP = sum(1 for a, b in zip(pred, truth) if a == 1 and b == 0)
    FN = sum(1 for a, b in zip(pred, truth) if a == 0 and b == 1)
    TN = sum(1 for a, b in zip(pred, truth) if a == 0 and b == 0)

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accu = (TP + TN) / (TP + FP + FN + TN) if TP + FP + FN + TN > 0 else 0.0

    return precision, recall, f1, accu


def calcu_confusion_matrix(pred, truth):
    return confusion_matrix(truth, pred)