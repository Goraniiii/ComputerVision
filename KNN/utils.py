
import numpy as np
import pandas as pd
import os

from PIL import Image


def load_cifar10_data(data_dir, num_samples=50000):
    print(f"Loading up to {num_samples} samples")

    labels_path = os.path.join(data_dir, 'trainLabels.csv')
    labels_df = pd.read_csv(labels_path)

    image_folder = os.path.join(data_dir, 'train')

    X_list = []
    y_list = []

    labels_to_load = labels_df.head(num_samples)

    for index, row in labels_to_load.iterrows():
        img_id = row['id']
        label = row['label']

        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        label_map = {name: i for i, name in enumerate(class_names)}
        int_label = label_map.get(label, -1)
        if int_label == -1:
             continue # skip undefined label

        # load image
        img_path = os.path.join(image_folder, f'{img_id}.png')
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_id}.png not found at {img_path}")
            continue

        try:
            img = Image.open(img_path)
            # image to numpy (32x32x3)
            img_array = np.array(img, dtype=np.float32)

            # image to 1-dimension vector (32*32*3 = 3072)
            X_vector = img_array.flatten() / 255.0

            X_list.append(X_vector)
            y_list.append(int_label)

        except Exception as e:
            print(f"Error loading image {img_id}: {e}")
            continue

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y

def split_data(X, y, split_ratio=0.8):
    num_data = X.shape[0]
    indices = np.arange(num_data)
    np.random.shuffle(indices)

    split_point = int(num_data * split_ratio)

    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    X_1 = X[train_indices]
    y_1 = y[train_indices]

    X_2 = X[test_indices]
    y_2 = y[test_indices]

    return X_1, y_1, X_2, y_2

def print_result(result):
    print(f"Accuracy: {result['accuracy']:.4f}, Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}, F1-score: {result['f1']:.4f}")


# data_dir = r'C:\Users\gony4\ComputerVision\KNN\CIFAR-10'
# NUM_SAMPLES = 50000
#
# X_full, y_full = load_cifar10_data(data_dir, num_samples=NUM_SAMPLES)
#
# TRAIN_DATA_RATIO = 0.9
#
# X_train, y_train, X_test, y_test = split_data(X_full, y_full, TRAIN_DATA_RATIO)
#
# print(f"Total training samples: {len(X_train)}")
# print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
#
#
# skknn = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
# skknn.fit(X_train, y_train)
# skknn.fit(X_train, y_train)
# y_sk_pred = skknn.predict(X_test)
#
# test_accuracy = np.mean(y_sk_pred == y_test)
# print(f"Test accuracy is {test_accuracy}")