import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
from data.load import FaceDataset


def train(model, criterion, optimizer, labeled_loader, unlabeled_loader, non_labelled_images,
          curr_labels, curr_images, non_labelled_image_deepface, train_transform,
          val_transform, device, max_epochs=50, stability_patience=5, early_stop_patience=10,
          selection_fraction=0.1, deepface_weight=0.7, model_weight=0.3, total_round_num=10):
    prev_val_f1 = None
    round_num = 0

    val_images = []
    val_labels = []
    total_train_loss = []
    total_val_loss = []

    model.train()
    while True:
        print(f"Round {round_num + 1}/{total_round_num} training started")
        best_val_loss = float('inf')
        no_improve_epochs = 0
        loss_list = []
        val_list = []
        for epoch in range(max_epochs):
            running_loss = 0.0
            for imgs, labels in labeled_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
            epoch_loss = running_loss / len(labeled_loader.dataset)
            loss_list.append(epoch_loss)
            print(f"Round {round_num} - Epoch {epoch}: Training Loss = {epoch_loss:.4f}")

        if val_images:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            print(f"val start: {len(val_images)}")
            with torch.no_grad():
                for i, img_path in enumerate(val_images):
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = val_transform(img).unsqueeze(0).to(device)
                    label = torch.tensor([val_labels[i]], dtype=torch.long).to(device)
                    output = model(img_tensor)
                    val_loss += criterion(output, label).item()
                    pred = output.argmax(dim=1)
                    correct += (pred == label).sum().item()
                    total += 1
            val_loss /= total
            acc = correct / total if total > 0 else 0.0
            print(f"Validation Loss = {val_loss:.4f}, Accuracy = {acc:.4f}")
            model.train()
            val_list.append(val_loss)
            # Early Stopping 判断
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                print("Early stopping triggered")
                break
        total_train_loss.append(loss_list)
        total_val_loss.append(val_list)

        cur_precision = cur_recall = cur_f1 = cur_acc = None
        if val_images:
            y_true = []
            y_pred = []
            y_score = []
            model.eval()
            with torch.no_grad():
                for i, img_path in enumerate(val_images):
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = val_transform(img).unsqueeze(0).to(device)
                    label = val_labels[i]
                    output = model(img_tensor)
                    probs = torch.softmax(output, dim=1)
                    prob1 = probs[0, 1].item()
                    pred_label = 1 if prob1 >= 0.5 else 0
                    y_true.append(label)
                    y_pred.append(pred_label)
                    y_score.append(prob1)
            model.train()
            correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
            cur_acc = correct / len(y_true) if y_true else 0.0
            # calculator Precision, Recall, F1
            TP = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
            FP = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
            FN = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
            precision = TP / (TP + FP) if TP + FP > 0 else 0.0
            recall = TP / (TP + FN) if TP + FN > 0 else 0.0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            print(
                f"Round {round_num} Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {cur_acc:.4f}")

            # dynamic selection threshold
            if prev_val_f1 is not None and cur_f1 is not None:
                if cur_f1 >= prev_val_f1:
                    # if performance improves, increase the selection fraction
                    selection_fraction = min(1.0, selection_fraction + 0.05)
                else:
                    # if performance decreased, decrease the selection fraction
                    selection_fraction = max(0.05, selection_fraction - 0.05)
            if cur_f1 is not None:
                prev_val_f1 = cur_f1

            if round_num >= 2:
                save_path = f"./model_weight/model_round_{round_num}.pth"
                torch.save(model.state_dict(), save_path)
                print(f"saved model weight: {save_path}")

            if len(non_labelled_images) == 0:
                print("if no unlabelled images -> break")
                break

            model.eval()
            new_train_samples = []
            new_val_samples = []
            high_confidence_records = []
            print(f"Round {round_num} - start selecting high confidence samples")
            with torch.no_grad():
                for imgs, img_paths in tqdm(unlabeled_loader):
                    imgs = imgs.to(device)
                    outputs = model(imgs)
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    deepface_probs = np.array([
                        non_labelled_image_deepface[path][1] / 100
                        for path in img_paths
                    ])
                    deepface_probs0 = 1 - deepface_probs
                    model_prob0 = 1 - probs
                    combined_prob1 = deepface_weight * deepface_probs + model_weight * probs
                    combined_prob0 = deepface_weight * deepface_probs0 + model_weight * model_prob0
                    high_confidence_records.extend(zip(img_paths, combined_prob1))

            model.train()
            combined_probs_sorted = sorted([prob for _, prob in high_confidence_records])

            if combined_probs_sorted:
                # select high confidence samples
                threshold_high = combined_probs_sorted[int((1 - selection_fraction) * len(combined_probs_sorted))]
                threshold_low = combined_probs_sorted[int(selection_fraction * len(combined_probs_sorted))]
            else:
                threshold_high, threshold_low = 0.5, 0.5

            if threshold_high >= 0.9:
                threshold_high = 0.9
            if threshold_low <= 0.1:
                threshold_low = 0.1

            if round_num >= total_round_num - 1:
                threshold_high, threshold_low = 0.5, 0.5

            added_count = 0
            removed_paths = []
            for path, prod in high_confidence_records:
                if prod >= threshold_high:
                    new_label = 1
                elif (1 - prod) <= threshold_low:
                    new_label = 0
                else:
                    continue  # 中间区域跳过
                new_train_samples.append((path, new_label))
                removed_paths.append(path)
                added_count += 1
            print(f"Round {round_num} - Added {added_count} samples to training set")

            non_labelled_images = [p for p in non_labelled_images if p not in removed_paths]
            unlabeled_dataset = FaceDataset(non_labelled_images, transform=val_transform)
            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=16, shuffle=False)

            if not val_images and len(new_train_samples) > 0:
                val_count = max(1, int(0.2 * len(new_train_samples)))
                val_subset = new_train_samples[:val_count]
                # update validation set
                for path, lbl in val_subset:
                    val_images.append(path)
                    val_labels.append(lbl)
                # update training set
                new_train_samples = new_train_samples[val_count:]
            elif len(new_train_samples) > 0:
                val_count = max(1, int(0.1 * len(new_train_samples)))
                val_subset = new_train_samples[:val_count]
                for path, lbl in val_subset:
                    val_images.append(path)
                    val_labels.append(lbl)
                new_train_samples = new_train_samples[val_count:]

            for path, lbl in new_train_samples:
                curr_images.append(path)
                curr_labels.append(lbl)

            labeled_dataset = FaceDataset(curr_images, labels=curr_labels, transform=train_transform)
            labeled_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)

            deepface_weight = max(0.0, deepface_weight - 0.1)
            model_weight = 1.0 - deepface_weight

            round_num += 1

            if round_num >= total_round_num:
                break
    return total_train_loss, total_val_loss
