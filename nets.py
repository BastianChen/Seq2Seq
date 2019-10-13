import torch.nn as nn
import torch


# 定义编码类
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(180, 256),
            nn.BatchNorm1d(256),
            nn.PReLU()
        )
        self.lstm = nn.LSTM(256, 128, 2, batch_first=True)

    def forward(self, data):
        # n,3,60,300 --> n,180,300 --> n,300,180
        data = data.reshape(data.size(0), 180, -1).permute(0, 2, 1)
        # n,300,180 --> n*300,180
        data = data.reshape(-1, 180)
        # n*300,180 --> n*300,256
        data = self.linear(data)
        # n*300,256 --> n,300,256
        data = data.reshape(-1, 300, 256)
        # n,300,256 --> n,300,128
        data, (h, c) = self.lstm(data, None)
        # 输出最后一个图像特征作为中间向量C
        # n,300,128 --> n,128
        output = data[:, -1, :]
        return output


# 定义节码类
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128, 256, 2, batch_first=True)
        self.linear = nn.Linear(256, 10)

    def forward(self, data):
        # n,128 --> n,1,128
        data = data.reshape(-1, 1, 128)
        # n,1,128 --> n,5,128 对s维度复制五份一样的值来预测验证码中的5个数字
        data = data.expand(-1, 5, 128)
        # n,5,128 --> n,5,256
        data, (h, c) = self.lstm(data)
        # n,5,256 --> n*5,256
        data = data.reshape(-1, 256)
        # n*5,256 --> n*5,10
        data = self.linear(data)
        # n*5,10 --> n,5,10
        output = data.reshape(-1, 5, 10)
        output = torch.softmax(output, 2)
        return output


# 定义一个总类调用编码类和解码类
class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, data):
        c = self.encoder(data)
        output = self.decoder(c)
        return output


if __name__ == '__main__':
    a = torch.Tensor(2, 1, 2)
    print(a)
    a = a.expand(2, 4, 2)
    print(a)
