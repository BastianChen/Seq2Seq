from nets import Seq2Seq
from dataset import Dataset
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


class Detector:
    def __init__(self, dataset_path, net_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.test_data = DataLoader(Dataset(dataset_path), batch_size=992, shuffle=True, drop_last=True)
        self.net = Seq2Seq().to(self.device)
        self.net.load_state_dict(torch.load(net_path))
        self.net.eval()

    def detect(self):
        for image_data, labels in self.test_data:
            image_data = image_data.to(self.device)
            labels = labels.to(self.device)
            output = self.net(image_data)
            output = torch.argmax(output, dim=2)
            labels = torch.argmax(labels, dim=2)
            acc = torch.sum(output == labels, dtype=torch.float32) / (self.test_data.batch_size * 5)
            print("acc:{}%".format(acc * 100))


if __name__ == '__main__':
    # detector = Detector("data/images/test", "models/net.pth")
    detector = Detector("data/images/test", "models/net_5.pth")
    detector.detect()
