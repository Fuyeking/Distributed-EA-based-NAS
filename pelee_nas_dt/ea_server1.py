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

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from deap import base, creator, tools
from dml import server_node as pn
from pelee_nas import ea_tools


class EaServerNode(pn.ServerNode):
    def __init__(self, ip_set, gen_l, pop, pop_size, gen_num):
        pn.ServerNode.__init__(self, ip_set, pop, pop_size)
        self.gen_num = gen_num
        self.gen_l = gen_l
        self.pop_size = pop_size
        self.gen_count = 0
        self.processed_data = []

    def controller(self, toolbox, cxpb, mutpb, block_size, func_list):

        for port, ip in self.ip_set.items():
            rec_pk = self.get_rec_data(port)
            if rec_pk is not None and rec_pk.op_type == 0:
                if self.send_data_left > 0:
                    if self.send_data_left >= rec_pk.ind_len:
                        data_len = rec_pk.ind_len
                    else:
                        data_len = self.send_data_left
                    self.add_send_data(port, self._create_packet(data_len))
                else:
                    self.add_send_data(port, self._create_packet(0))

            if rec_pk is not None and rec_pk.op_type == 1:
                print("port, handled data", port, rec_pk.handle_len, rec_pk.ind_set)
                self.rec_count += rec_pk.handle_len
                self.processed_data.extend(rec_pk.ind_set)
                if self.rec_count == self.pop_size and self.gen_count < self.gen_num:
                    print("evolution:", self.processed_data)
                    population = self.ea_simple(self.processed_data, toolbox, cxpb, mutpb, block_size, func_list)
                    self.sharing_send_data = population
                    self.gen_count += 1
                    self.send_data_left = len(population)
                    # 清空数据，重新开始
                    self.rec_count = 0
                    self.processed_data = []
                # 重新分配数据
                if self.send_data_left > 0:
                    if self.send_data_left >= rec_pk.ind_len:
                        data_len = rec_pk.ind_len
                    else:
                        data_len = self.send_data_left
                    self.add_send_data(port, self._create_packet(data_len))
                else:
                    if self.gen_count == self.gen_num and self.rec_count == self.pop_size:
                        self.termination = True
                    data_len = 0
                    self.add_send_data(port, self._create_packet(data_len))

        if self.termination:
            for port, ip in self.ip_set.items():
                self.add_send_data(port, self._create_packet(0))

    def ea_simple(self, population, toolbox, cxpb, mutpb, block_size, func):
        offspring = toolbox.select(population, len(population))
        # Vary the pool of individuals
        offspring = ea_tools.varAnd(offspring, toolbox, cxpb, mutpb, self.gen_l, block_size, func)
        population[:] = offspring
        return population


# Hyper Parameters

POP_SIZE = 2  # 种群大小
GEN_NUM = 2  # 演化的代数
GEN_L = 3  # 基因的长度
BLOCK_STAGE = 4

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
func_seq = [lambda: random.randint(1, 3), lambda: random.randint(1, 10), lambda: random.randint(1, 2)]
toolbox.register("individual", tools.initCycle, creator.Individual, func_seq, BLOCK_STAGE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selRoulette)
pop = toolbox.population(n=POP_SIZE)


# In[ ]:
def main():
    # ip_sets = {12346: "127.0.0.1", 12347: "127.0.0.1", 12348: "127.0.0.1"}  # , 12349: "127.0.0.1"}
    ip_sets = {12345: "127.0.0.1"}
    mp.set_start_method('spawn')
    # ip_sets = {12345: "127.0.0.1"}
    master_node = EaServerNode(ip_sets, GEN_L, pop, POP_SIZE, GEN_NUM)
    master_node.distributed_dnn()
    start = time.process_time()
    while True:
        master_node.controller(toolbox, cxpb=0.5, mutpb=0.5, block_size=BLOCK_STAGE, func_list=func_seq)
        if master_node.termination:
            # print top-3 optimal solutions
            best_individuals = tools.selBest(pop, k=3)
            for bi in best_individuals:
                print(bi)
            break
    elapsed = (time.process_time() - start)
    print("elapsed time:%.3f" % (elapsed / (60 * 60.0)))
    master_node.close_process()


if __name__ == '__main__':
    main()

# --------------------------------------------------------------------------------------------------------------------
