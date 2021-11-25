# !/usr/bin/python
# -*- coding: UTF-8 -*-
import multiprocessing as mp
import socket

import dml.dml_work_process as dmt
from dml import packet

data_size = 1024


class WorkerNode:

    def __init__(self, worker_name):
        """
        """
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.net_ready = False  # 网络连接准备状态
        self.send_queue = mp.Queue()
        self.rec_queue = mp.Queue()
        self.socket_reference_count = 0
        self.send_process = None
        self.rec_process = None
        self.name = worker_name

    def __del__(self):
        self.__close_socket()

    def increase_socket_reference_count(self):
        self.socket_reference_count += 1

    def decrease_socket_reference_count(self):
        self.socket_reference_count -= 1

    def connect(self, host: object, port: object) -> object:
        self.server_socket.connect((host, port))
        return self.server_socket

    def prepare_net(self) -> object:
        while not self.net_ready:
            data = self.server_socket.recv(data_size)
            if data.decode("utf-8") == "OK":
                self.net_ready = True

    def init_process(self):
        self.rec_process = dmt.WorkBaseRecProcess(self.name, self)
        self.send_process = dmt.WorkBaseSendProcess(self.name, self)

    def start_process(self):
        self.rec_process.start()
        self.send_process.start()

    def stop_process(self):
        self.rec_process.terminate()
        self.send_process.terminate()

    def send_new_packet(self, data_len):
        req_pk = packet.Packet()
        req_pk.ind_len = data_len
        self.add_send_data(req_pk)

    def send_reply_packet(self, rp_data, data_len, handle_len):
        rp_pk = packet.Packet()
        rp_pk.ind_set = rp_data
        rp_pk.ind_len = data_len
        rp_pk.handle_len = handle_len
        rp_pk.op_type = 1
        self.add_send_data(rp_pk)

    def add_send_data(self, data):
        self.send_queue.put(data)

    def get_rec_data(self):
        if not self.rec_queue.empty():
            data = self.rec_queue.get()
            if data is not None:
                return data

    def __close_socket(self):
        if self.socket_reference_count == 0:
            self.server_socket.close()
