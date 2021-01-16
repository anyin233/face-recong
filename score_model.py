from model.multi_model import ConcatModel
from model.vgg16 import Vgg16
from model.ResNet import ResNet50

from utils.dataset import FaceDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import time

dataset = FaceDataset("data/face_info.csv", train=False)
loader = DataLoader(dataset, batch_size=1)

concat = ConcatModel(2)
vgg = Vgg16(2)
resnet = ResNet50(2)
concat_dict = torch.load("deploy_model/concat")
vgg_dict = torch.load("deploy_model/vgg")
resnet_dict = torch.load("deploy_model/resnet")
concat.load_state_dict(concat_dict)
vgg.load_state_dict(vgg_dict)
resnet.load_state_dict(resnet_dict)
concat = concat.cuda()
vgg = vgg.cuda()
resnet = resnet.cuda()
concat_count = 0
vgg_count = 0
resnet_count = 0
count = 0

start_time = time.time()

for img, label in tqdm(loader):
    img = img.cuda()
    label = label.cuda()
    concat.eval()
    vgg.eval()
    resnet.eval()
    _, concat_res = concat(img).max(1)
    _, vgg_res = vgg(img).max(1)
    _, resnet_res = resnet(img).max(1)
    if concat_res == label:
        concat_count += 1
    if vgg_res == label:
        vgg_count += 1
    if resnet_res == label:
        resnet_count += 1
    count += 1

print("acc: concat:{}\tvgg:{}\tresnet:{}\ncost {}".format(concat_count/count, vgg_count/count, resnet_count/count, time.time() - start_time))
