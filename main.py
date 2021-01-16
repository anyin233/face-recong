import torch
import torch.optim as optim
from model.ResNet import ResNet50
from model.vgg16 import Vgg16
from model.multi_model import ConcatModel
import torch.distributed as distributed
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from utils.dataset import FaceDataset
from tensorboardX import SummaryWriter
import pickle
import time
import os

start_time = int(time.time())
time_local = time.localtime(start_time)
dt = time.strftime("%Y-%m-%d-%H-%M-%S", time_local)
cuda0 = torch.device("cuda:0")
cuda1 = torch.device("cuda:1")


def train(model_name, train_dataset, test_dataset, lr=2e-3, epoches=200, batch_size=64, cuda=True):
    # device = torch.device('cuda:1') if cuda else torch.device('cpu')
    # distributed.init_process_group(backend='nccl', init_method='tcp://localhost:12345', rank=0, world_size=1)
    # model = ResNet50(3)
    # model = Vgg16(2)
    if model_name == 'Concat':
        model = ConcatModel(2)
    elif model_name == 'VGG':
        model = Vgg16(2)
    elif model_name == 'ResNet':
        model = ResNet50(2)
    else:
        raise NotImplemented
    model = DataParallel(model)
    model = model.cuda()
    writer = SummaryWriter(logdir='log/{}'.format(dt))
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    flag = True
    for ep in range(epoches):
        total_loss = 0
        count = 1
        model.train()
        for image, label in train_loader:
            image, label = image.cuda(), label.cuda()
            if flag:
                # writer.add_graph(model, input_to_model=image)
                flag = False
            if ep % 100 == 0:
                writer.add_images("{}_train_images".format(model_name), image, ep)
            output = model(image)
            optimizer.zero_grad()
            loss = loss_func(output, label)
            loss.backward()
            total_loss += loss.item()
            count += 1
            optimizer.step()
        current_cuda0 = torch.cuda.memory_allocated(cuda0)
        current_cuda1 = torch.cuda.memory_allocated(cuda1)
        total = (current_cuda1 + current_cuda0) / (1024 ** 3)
        writer.add_scalar("{}_memory_allocated".format(model_name), total, ep)
        writer.add_scalar("{}_train_loss".format(model_name), total_loss / count, global_step=ep)
        print("ep: {}, loss: {}".format(ep + 1, total_loss / count))
        if (ep + 1) % 10 == 0:
            print("testing model")
            test_loss = 0
            test_count = 1
            model.eval()
            for image, label in test_loader:
                image, label = image.cuda(), label.cuda()
                output = model(image)
                loss = loss_func(output, label)
                test_loss += loss.item()
                test_count += 1
            path_name = "saved_model/{}".format(dt)
            if not os.path.exists(path_name):
                os.mkdir(path_name)
            model_path = "saved_model/{}/{}/ep{}-loss{:.4}".format(dt, model_name, ep + 1, test_loss / test_count)
            print("test result: {}@ep{}, loss:{}\nsaving model to {}".format(model_name, ep + 1, test_loss / test_count, model_name))
            writer.add_scalar('{}_test_loss'.format(model_name), test_loss / test_count, ep)
            # pickle.dump(model.state_dict(), open(model_name, 'wb'))
            torch.save(model.module.state_dict(), model_path)
    writer.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    if not os.path.exists("saved_model/{}".format(dt)):
        os.mkdir("saved_model/{}".format(dt))
    if not os.path.exists("saved_model/{}/{}".format(dt, "Concat")):
        os.mkdir("saved_model/{}/{}".format(dt, "Concat"))
    if not os.path.exists("saved_model/{}/{}".format(dt, "VGG")):
        os.mkdir("saved_model/{}/{}".format(dt, "VGG"))
    if not os.path.exists("saved_model/{}/{}".format(dt, "ResNet")):
        os.mkdir("saved_model/{}/{}".format(dt, "ResNet"))
    train("Concat", FaceDataset('data/face_info.csv'), FaceDataset('data/face_info.csv', train=False))
    train("VGG", FaceDataset('data/face_info.csv'), FaceDataset('data/face_info.csv', train=False), epoches=300)
    train("ResNet", FaceDataset('data/face_info.csv'), FaceDataset('data/face_info.csv', train=False), epoches=300)
