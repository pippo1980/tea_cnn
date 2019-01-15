import os
import os.path as path

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable

import numpy as np
from PIL import Image

import extract.image_part as part

samples_root = path.join(os.getcwd(), "../samples")
features_root = path.join(os.getcwd(), "../features")

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # 正则化
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
)


def make_model():
    model = models.resnet50(pretrained=True)

    # 将模型从CPU发送到GPU
    if torch.cuda.is_available():
        model.cuda()

    model.fc = nn.Linear(2048, 2048)

    torch.nn.init.eye_(model.fc.weight)
    for param in model.parameters():
        param.requires_grad = False

    return model


def file_extractor(file_path, feature_path, net, use_gpu):
    img = Image.open(file_path)
    img = transform(img)

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    if use_gpu:
        x = x.cuda()
        net = net.cuda()

    y = net(x).cpu()
    y = y.data.numpy()
    np.savetxt(feature_path, y, delimiter=',')


def dir_extractor(_image_dir, _feature_dir, net, use_gpu=False):
    if not os.path.exists(_feature_dir):
        os.makedirs(_feature_dir)

    for file in os.listdir(_image_dir):

        if not file.endswith("jpg"):
            continue

        image_path = _image_dir + "/" + file
        feature_path = _feature_dir + "/" + file + ".txt"
        print(file, " to ", feature_path)

        # file_extractor(image_path, feature_path, net, use_gpu)

        file_extractor(image_path, feature_path, net, use_gpu)


if __name__ == '__main__':
    print(os.getcwd())
    sample_dir = samples_root + "/20190112"
    for file in os.listdir(sample_dir):
        if not file.endswith("jpg"):
            continue

        base_image = sample_dir + "/" + file
        part.image_parts(base_image, sample_dir + "/parts", file)

    feature_dir = features_root + "/20190112"

    dir_extractor(sample_dir + "/parts", feature_dir, make_model(), False)
