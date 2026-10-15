import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch import nn, optim

# Settings
data_dir = 'D:\Image Stuff\DATASET'
num_classes = 4
batch_size = 12
epochs = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Datasets
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model (pretrained)
model = models.resnet50(pretrained=True)  # or vit_b_16 for Vision Transformer
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

from tqdm import tqdm

# Training loop with progress bar
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

    print(f"Epoch {epoch + 1} completed.")
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved as model.pth!")

print("Training done!")