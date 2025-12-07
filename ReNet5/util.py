import torch
import torch.nn.functional as F
import numpy as np
import cv2

import matplotlib.pyplot as plt


def train_rbf(model, device, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        labels_onehot = F.one_hot(labels, num_classes=4).float().to(device)
        loss = criterion(outputs, labels_onehot)
        loss.backward()
        optimizer.step()

        # log
        running_loss += loss.item()

        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch} - Loss: {running_loss/len(loader):.4f}, Train Acc: {acc:.4f}")

def train(model, device, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # log
        running_loss += loss.item()

        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch} - Loss: {running_loss/len(loader):.4f}, Train Acc: {acc:.4f}")


def evaluate_rbf(model, device, loader, criterion):
    model.eval()

    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            labels_onehot = F.one_hot(labels, num_classes=4).float().to(device)
            loss = criterion(outputs, labels_onehot)

            test_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Test Loss: {test_loss/len(loader):.4f}, Test Acc: {acc:.4f}")
    return acc


def evaluate(model, device, loader, criterion):
    model.eval()

    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Test Loss: {test_loss/len(loader):.4f}, Test Acc: {acc:.4f}")
    return acc

def show_samples(dataset, samples_per_class=5):
    class_images = {i: [] for i in range(4)}

    # collect
    for img, label in dataset:
        label = int(label)

        if len(class_images[label]) < samples_per_class:
            class_images[label].append(img)

    fig, axes = plt.subplots(4, samples_per_class, figsize=(12, 10))

    for cls_idx in range(4):
        for i in range(samples_per_class):
            img = class_images[cls_idx][i]

            # tensor → numpy
            img_np = img.numpy()

            # (C,H,W) → (H,W,C)
            if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
                img_np = np.transpose(img_np, (1, 2, 0))

            # grayscale (H,W,1) → (H,W)
            if img_np.shape[-1] == 1:
                img_np = img_np.squeeze(-1)

            ax = axes[cls_idx, i]
            ax.imshow(img_np, cmap="gray")
            ax.axis('off')

    plt.tight_layout()
    plt.show()

class ToBinary:
    def __init__(self):
        return

    def __call__(self, image):
        # image: PIL Image → numpy array 변환
        image = np.array(image)

        # 1) RGB → Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 2) OTSU threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh  # numpy array (H,W)

class Resize:
    def __init__(self, size=(32, 32)):
        self.size = size

    def __call__(self, image):
        # image: PIL Image → numpy array 변환
        image = np.array(image)

        # 3) Resize
        resized = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)

        return resized  # numpy array (H,W)