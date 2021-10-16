# -*- coding:utf-8 -*-
from Trainer import Net,USE_DEFAULT_MODEL
import torch
from torch.autograd import Variable

##################################################
# change torch model into onnx
##################################################
if __name__ == "__main__":
    # load net shape and train weight
    if USE_DEFAULT_MODEL:
        from torchvision.models.resnet import resnet18
        trained_model = resnet18()
    else:
        trained_model = Net()
    trained_model.load_state_dict(torch.load('output/mymodel.pth'))
    # prepare dummy input (this dummy input shape should same with train input)
    dummy_input = Variable(torch.randn(1, 3, 224, 224))
    # export onnx model
    torch.onnx.export(trained_model, dummy_input, "output/mymodel.onnx")
