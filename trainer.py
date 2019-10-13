from nets import Seq2Seq
from dataset import Dataset
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os


class Trainer:
    def __init__(self, dataset_path, net_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_path = net_path
        self.dataset = Dataset(dataset_path)
        self.train_data = DataLoader(self.dataset, batch_size=600, shuffle=True, drop_last=True)
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.net = Seq2Seq().to(self.device)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())
        if os.path.exists(net_path):
            self.net.load_state_dict((torch.load(net_path)))
        self.net.train()

    def train(self):
        epoch = 1
        loss_new = 100
        while True:
            for i, (image_data, labels) in enumerate(self.train_data):
                image_data = image_data.to(self.device)
                labels = labels.to(self.device)
                output = self.net(image_data)
                loss = self.loss(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # output = torch.argmax(output, dim=2)
                # labels = torch.argmax(labels, dim=2)
                # acc = torch.sum(output == labels, dtype=torch.float32) / (self.train_data.batch_size * 5)
                # print("epoch:{},i:{},loss:{:.10f},acc:{}%".format(epoch, i, loss.item(), acc * 100))

                if i % 10 == 0:
                    output = torch.argmax(output, dim=2)
                    labels = torch.argmax(labels, dim=2)
                    acc = torch.sum(output == labels, dtype=torch.float32) / (self.train_data.batch_size * 5)
                    print("epoch:{},i:{},loss:{:.10f},acc:{}%".format(epoch, i, loss.item(), acc * 100))
                if loss.item() < loss_new:
                    loss_new = loss.item()
                    torch.save(self.net.state_dict(), self.net_path)
            epoch += 1
