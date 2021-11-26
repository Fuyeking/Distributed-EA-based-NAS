#!/usr/bin/env python
# encoding: utf-8
'''
@author: yeqing
@contact: 474387803@qq.com
@software: pycharm
@file: ea_cnn.py
@time: 2019/6/16 16:16
@desc:
'''
import multiprocessing as mp
import os
import random
import sys
import time

import torch.cuda
import torch.nn as nn
import torch.utils.data as DataUtil
import torchvision
import torchvision.transforms as transforms
from deap import base, creator, tools

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dml import worker_node as cn
from pelee_nas import ea_tools
from pelee_nas.pelee import PeleeNet as Pelee

module = __import__("dml.dml_work_process")
# Hyper Parameters
EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 256
LR = 0.18  # learning rate
Momentum = 0.9
Weight_Decay = 1e-4
NUM_CLASS = 10
GEN_L = 3  # 每一段基因的长度
BLOCK_STAGE = 4
growth_step = [8, 16, 32]
Input_dim = 64
# 训练用的数据集
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_data = torchvision.datasets.CIFAR10(root="data",
                                          train=True,
                                          transform=transforms.Compose([
                                              transforms.RandomResizedCrop(Input_dim),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]),
                                          target_transform=None,
                                          download=True)
train_loader = DataUtil.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = torchvision.datasets.CIFAR10(root="data", train=False,
                                         transform=transforms.Compose([
                                             transforms.Resize(Input_dim),
                                             transforms.CenterCrop(Input_dim),
                                             transforms.ToTensor(),
                                             normalize,
                                         ]),
                                         target_transform=None,
                                         download=True)
test_loader = DataUtil.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


def prep_hyper_parameters(individual, growth_steps, gen_l):
    growth_rate = []
    block_config = []
    dense_way = []
    gen_set = ea_tools.split_gen(individual, gen_l)
    for gen in gen_set:
        growth_rate.append(gen[0])
        block_config.append(gen[1])
        dense_way.append(gen[2])
    for i in range(len(growth_rate)):
        growth_rate[i] = growth_steps[growth_rate[i] % gen_l]
    return growth_rate, block_config, dense_way


# 计算网络模型
def calc_model_size(individual, growth_steps, gen_l):
    growth_rate, block_config, dense_way = prep_hyper_parameters(individual, growth_steps, gen_l)
    model = Pelee(growth_rate=growth_rate, block_config=block_config, dense_nums=None, num_classes=NUM_CLASS).to(
        device)
    return ea_tools.summary(model, input_size=(3, Input_dim, Input_dim))


# 模型测试获得模型的准确率
def test_model(model):
    total = 0
    correct = 0.0
    for step, (images, labels) in enumerate(
            test_loader):  # gives batch data, normalize x when iterate train_loader
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        # 取得最高分的那个类（outputs.data的索引号
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return 100 * correct / total


# 模型的评估
def evaluate_model(individual):
    growth_rate, block_config, dense_way = prep_hyper_parameters(individual, growth_step, GEN_L)
    model = Pelee(growth_rate=growth_rate, block_config=block_config, dense_nums=None, num_classes=NUM_CLASS).to(
        device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # optimize all cnn parameters
    optimizer = torch.optim.SGD(model.parameters(), LR,
                                momentum=Momentum,
                                weight_decay=Weight_Decay)
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    time_step = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    file_name = "log3_1" + str(individual) + ".txt"
    fp = open(file_name, "w")
    # training and testing
    start = time.process_time()
    for epoch in range(EPOCH):
        for step, (images, labels) in enumerate(
                train_loader):  # gives batch data, normalize x when iterate train_loader
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # cnn output
            loss = loss_func(outputs, labels)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()
        # For Test
        accuracy = test_model(model)
        print('epoch%3d,accuracy:%.3f%%' % (epoch + 1, accuracy))
    elapsed = (time.process_time() - start)
    model_size = ea_tools.summary(model, input_size=(3, Input_dim, Input_dim))
    fp.write('epoch%3d,accuracy:%.3f%%' % (epoch + 1, accuracy))
    fp.write('\n')
    fp.write("size of mode%.3fM" % model_size)
    fp.flush()
    fp.close()
    print("Size of mode %.3fM|elapsed time:%.3f M" % (model_size, elapsed / 60.0))
    return accuracy.to("cpu"),


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
func_seq = [lambda: random.randint(1, 3), lambda: random.randint(1, 10), lambda: random.randint(1, 2)]
toolbox.register("individual", tools.initCycle, creator.Individual, func_seq, BLOCK_STAGE)
toolbox.register("evaluate", evaluate_model)
port = 12345
ip = "127.0.0.1"

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    worker = cn.WorkerNode("worker2")
    worker.connect(ip, port)
    worker.prepare_net()
    worker.init_process()
    worker.start_process()
    worker.send_new_packet(2)
    end = False
    while True:
        rec_pk = worker.get_rec_data()
        if rec_pk is not None:
            if rec_pk.termination:
                break
            if rec_pk.ind_len > 0:
                print("rec data:", rec_pk.ind_set)
                # 数据处理
                pop = rec_pk.ind_set
                result = ea_tools.dis_ea_simple(pop, toolbox)
                worker.send_reply_packet(result, 1, len(result))
            else:
                worker.send_new_packet(1)
    print("main process end")
    worker.stop_process()
