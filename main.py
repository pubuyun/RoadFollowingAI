import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
#------------------------------------DATASET------------------------------------#
# 定义自定义数据集
WIDTH = 360
HEIGHT = 360
TRAINING_FILE_PATH = 'train/dataset.csv'
class AutoDriveDataset(Dataset):
    def __init__(self, csv_file):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.Resize((WIDTH, HEIGHT)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        image = Image.open(img_path)
        image = self.transform(image)
        angle = self.dataframe.iloc[idx, 1]
        speed = self.dataframe.iloc[idx, 2]
        direct = self.dataframe.iloc[idx, 3]
        sample = {'image': image, 'angle': angle, 'speed': speed, 'direct': direct}
        return sample
    
#------------------------------------MODEL------------------------------------#
# 创建数据加载器
dataset = AutoDriveDataset(csv_file=TRAINING_FILE_PATH)
trainloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
class AutoDriveModel(nn.Module):
    def __init__(self):
        super(AutoDriveModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=12, stride=6, padding=0),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.rnn = nn.LSTM(input_size=25600, hidden_size=320, num_layers=1, batch_first=True)
        self.linear = nn.Linear(320, 2)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        timesteps = 1  # Set timesteps to 1
        cnn_in = x.view(batch_size * timesteps, C, H, W)
        cnn_out = self.cnn(cnn_in)
        rnn_in = cnn_out.view(batch_size, timesteps, -1)
        rnn_out, _ = self.rnn(rnn_in)
        output = self.linear(rnn_out[:, -1, :])
        return output

model = AutoDriveModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
#------------------------------------TRAIN------------------------------------#
# Training loop
for epoch in range(128):
    for i, data in enumerate(trainloader, 0):
        # Correctly unpack the data
        inputs = data['image']
        angle_labels = data['angle']
        speed_labels = data['speed']
        direct_labels = data['direct']
        
        optimizer.zero_grad()
        
        # Model forward pass
        outputs = model(inputs)
        
        # Split the output into angle and speed predictions
        angle_outputs, speed_outputs = outputs[:, :1], outputs[:, 1:]
        
        # Calculate the loss for angle and speed predictions
        angle_loss = criterion(angle_outputs, angle_labels.float())
        speed_loss = criterion(speed_outputs, speed_labels.float())
        loss = angle_loss + speed_loss
        loss = loss.float()
        print(loss)
        
        loss.backward()
        optimizer.step()

#------------------------------------TEST------------------------------------#
transform = transforms.Compose([
            transforms.Resize((WIDTH, HEIGHT)),
            transforms.ToTensor(),
        ])
image = Image.open('resources/1.jpg')
image = transform(image)
outputs = model(torch.unsqueeze(image,0))
outputs