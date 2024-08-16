import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
from sklearn.metrics import confusion_matrix, classification_report
from facenet_pytorch import InceptionResnetV1
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Adjusted parameters for InceptionResnetV1
IMAGE_SIZE = 128
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 1  # Binary classification
EPOCHS = 150

def read_and_preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img, dtype='float32')
    img = img / 255
    return img

labels = ['real', 'fake']

X = []  # To store images
y = []  # To store labels

# labels -
# 0 - Real
# 1 - Fake
image_path = 'C:/Users/roshi/PycharmProjects/DEEPTRUTH/Model Creation/dataset/'

for folder in os.scandir(image_path):
    for entry in os.scandir(image_path + folder.name):
        X.append(read_and_preprocess(image_path + folder.name + '/' + entry.name))

        if folder.name[0] == 'r':
            y.append(0)  # real
        else:
            y.append(1)  # fake

X = np.array(X)
y = np.array(y)

real_count = len(y[y == 0])
fake_count = len(y[y == 1])

plt.title("Train Images for Each Label")
plt.bar(["Real Images", "Fake Images"], [real_count, fake_count])
device = "cuda" if tf.test.is_gpu_available() else "cpu"

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=123)
X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.5, shuffle=True, stratify=y_val, random_state=123)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create PyTorch DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# InceptionResnetV1 Model
base_model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=1,
                               device='cuda' if torch.cuda.is_available() else 'cpu')

# Customize the model head
x = base_model.logits.in_features
model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=1,
                          device='cuda' if torch.cuda.is_available() else 'cpu')

model.logits = nn.Linear(x, 1)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # sigmoid + binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels.unsqueeze(1).float())

        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()

            predicted = torch.round(torch.sigmoid(outputs)).squeeze().cpu().numpy().astype(int)
            correct += (predicted == labels.cpu().numpy()).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Save the model
torch.save(model.state_dict(), "resnetinceptionv1_final.pth")

# Evaluate on the test set
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.round(torch.sigmoid(outputs)).squeeze().cpu().numpy().astype(int)
        predictions.extend(predicted)
        true_labels.extend(labels.cpu().numpy())

# Confusion Matrix
cf_matrix = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(9, 7))
group_names = ['Real Images Predicted Correctly', 'Real Images Predicted as Fake', 'Fake Images Predicted as Real',
               'Fake Images Predicted Correctly']
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

# Classification Report
report = classification_report(true_labels, predictions)
print(report)
