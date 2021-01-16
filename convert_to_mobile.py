import torch.quantization as quantization
import torch
import argparse
from model.multi_model import ConcatModel
from model.ResNet import ResNet50
from model.vgg16 import Vgg16
import torch.utils.mobile_optimizer as mobile_optimizer


def load_model(path):
    model = torch.load(path, map_location=torch.device("cpu"))
    return model


def convert(model, path):
    state_dict = load_model(path)
    model.load_state_dict(state_dict)
    model.eval()
    opt_model = quantization.convert(model)
    return opt_model


def convert_to_mobile(model_name, path, dest):
    if model_name == 'concat':
        model = ConcatModel(2)
    elif model_name == 'vgg':
        model = Vgg16(2)
    else:
        model = ResNet50(2)
    model = convert(model, path)
    script_model = torch.jit.script(model)
    mobile_model = mobile_optimizer.optimize_for_mobile(script_model)
    torch.jit.save(mobile_model, dest)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--model", dest="model", type=str, choices=['concat', 'vgg', 'resnet'])
    args.add_argument("-p", "--path", dest="path", type=str)
    args.add_argument("-d", "--dest", dest="dest", type=str)
    opts = args.parse_args()
    convert_to_mobile(opts.model, opts.path, opts.dest)
