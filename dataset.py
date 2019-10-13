from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np


class Dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.datasets = os.listdir(path)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, item):
        image_name = self.datasets[item]
        image_data = self.trans(Image.open(os.path.join(self.path, image_name)))
        label = image_name.split(".")[0]
        label = self.ont_hot(label)
        return image_data, label

    def ont_hot(self, label):
        # 类型为float是为了防止后面做损失时输出值为float类型与integer类型的标签不匹配
        result = np.zeros([5, 10], dtype=np.float32)
        for i in range(5):
            index = label[i]
            result[i][int(index)] = 1.
        return result


if __name__ == '__main__':
    dataset = Dataset("data/images/test")
    image_data, label = dataset[1]
    print(image_data.shape)
    print(label)
    print(label.shape)
