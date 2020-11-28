import cv2 as cv
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class Video:
    def __init__(self, filename):
        self.cap = cv.VideoCapture(filename)

    def __getitem__(self, i):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, i)
        ret, frame = self.cap.read()

        if ret:
            return frame
        else:
            exit('error')

class TrainingDataset(Dataset):
    def __init__(self):
        self.video = Video('data/train.mp4')
        
        with open('data/train.txt') as f:
            self.speeds = tuple(map(float, f.read().split('\n')))

    def __len__(self):
        return 20

    def __getitem__(self, i):
        return self.video[i].flatten() / 255, torch.tensor([self.speeds[i]], dtype=torch.double)

    def frame_size(self):
        return self.video[0].size

training_dataset = TrainingDataset()
training_data_loader = DataLoader(training_dataset, len(training_dataset))
net = nn.Linear(training_dataset.frame_size(), 1).double()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), 1e-5)
losses = []

for i in range(100):
    for frames, speeds in training_data_loader:
        optimizer.zero_grad()

        loss = criterion(net(frames), speeds)

        print('epoch ' + str(i + 1) + ' loss: ' + str(loss.item()))

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(losses)
plt.show()
