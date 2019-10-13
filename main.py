from trainer import Trainer

if __name__ == '__main__':
    # trainer = Trainer("data/images/train", "models/net.pth")
    trainer = Trainer("data/images/train", "models/net_5.pth")
    trainer.train()
