import torch
from torch.utils.data import DataLoader

class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNClassifier(torch.nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # (batch_size, 32, 28, 28)
        x = self.pool(x)  # (batch_size, 32, 14, 14)
        x = torch.relu(self.conv2(x))  # (batch_size, 64, 14, 14)
        x = self.pool(x)  # (batch_size, 64, 7, 7)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))  # (batch_size, 128)
        x = self.fc2(x)  # (batch_size, num_classes)
        return x

def train_mnist_classifier(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    # prints training accuracy for each epoch
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    print(f'Training completed for {num_epochs} epochs.')

if __name__ == "__main__":
    # Let's define train_loader here using torchvision.datasets.MNIST
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    # separate training and validation dataset
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

    # Example usage
    input_dim = 784  # For MNIST images (28x28)
    hidden_dim = 128
    output_dim = 10  # Number of classes in MNIST

    model = LinearClassifier(input_dim, hidden_dim, output_dim)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train_loader is defined elsewhere and provides MNIST data
    num_epochs = 5
    train_mnist_classifier(model, train_loader, criterion, optimizer, num_epochs)

    # validate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Validation Accuracy: {100 * correct / total:.2f}%')
