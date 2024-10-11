import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from hybrid_network import HybridNet  # 确保模型导入正确
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processed_image_data_array = np.load('finally_process.npy')
processed_image_data_array = processed_image_data_array[:, np.newaxis, :, :, :]

with open('labels.txt', 'r') as file:
    labels_str = file.read().split()
    labels = list(map(int, labels_str))
labels = torch.LongTensor(labels)

# split
X_train, X_test, y_train, y_test = train_test_split(processed_image_data_array, labels, test_size=0.3, shuffle=True)


# CustomDataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return image, label


batch_size = 146
train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

conv_input_channel = 30
conv_output_channel = 576
num_primary_node = 32 * 6 * 6
primary_node_size = 8
output_node_size = 16
num_classes = 3
learning_rate = 0.01
early_stop_loss = 0.0001

# model
network = HybridNet(
    image_width=28,
    image_height=28,
    image_channels=30,
    conv_input_channel=conv_input_channel,
    conv_output_channel=conv_output_channel,
    num_primary_node=num_primary_node,
    primary_node_size=primary_node_size,
    num_output_node=num_classes,
    output_node_size=output_node_size
).to(device)

optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=1e-4)


# one-hot
def to_one_hot(x, num_classes):
    x = torch.clamp(x, 0, num_classes - 1)
    batch_size = x.size(0)
    x_one_hot = torch.zeros(batch_size, num_classes).to(device)
    x_one_hot.scatter_(1, x.view(-1, 1), 1)
    return x_one_hot


def train(epoch):
    network.train()
    last_loss = None
    log_interval = 10
    for batch_idx, (data, target) in enumerate(train_loader):

        target_one_hot = to_one_hot(target, num_classes=num_classes)
        data, target = data.to(device).float(), target_one_hot.to(device).float()

        optimizer.zero_grad()
        output = network(data)
        loss = network.loss(data, output, target)
        loss.backward()
        optimizer.step()

        last_loss = loss.item()

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        if last_loss < early_stop_loss:
            break

    return last_loss


num_epochs = 20
for epoch in range(1, num_epochs + 1):
    last_loss = train(epoch)

    if last_loss < early_stop_loss:
        print(f"Early stopping at epoch {epoch} due to low loss: {last_loss}")
        break
