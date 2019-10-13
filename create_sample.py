from PIL import Image, ImageDraw, ImageFont
import numpy as np


class Create:
    def __init__(self, number, save_path):
        self.number = number
        self.save_path = save_path
        self.w = 300
        self.h = 60
        self.font = ImageFont.truetype("/data/font/simkai.ttf", 40)

    def getRandomChr(self):
        return chr(np.random.randint(48, 58))

    def getRandomColor(self):
        background_color = (np.random.randint(100, 256), np.random.randint(100, 256), np.random.randint(100, 256))
        text_color = (np.random.randint(30, 131), np.random.randint(30, 131), np.random.randint(30, 131))
        return [background_color, text_color]

    def createSample(self):
        for i in range(self.number):
            image = Image.new("RGB", (self.w, self.h), (255, 255, 255))
            draw = ImageDraw.ImageDraw(image)
            for x in range(self.w):
                for y in range(self.h):
                    draw.point((x, y), fill=self.getRandomColor()[0])
            filename = ""
            for j in range(5):
                text = self.getRandomChr()
                draw.text((60 * j + 20, 10), text=text, font=self.font, fill=self.getRandomColor()[1])
                filename += text
            image.save("{}/{}.jpg".format(self.save_path, filename))


if __name__ == '__main__':
    create_train = Create(50000, "data/images/train")
    create_test = Create(1000, "data/images/test")
    create_train.createSample()
    create_test.createSample()
