import pickle
from multiprocessing import Process

from dml import server_proxy as sn


# 服务端的接受线程
class ServerRecBaseProcess(Process):

    def __init__(self, thread_name):
        super(ServerRecBaseProcess, self).__init__()
        self.thread_name = thread_name
        self.server_obj = None
        self.rec_queue = None

    def init_para(self, server_obj, rec_q):
        self.server_obj = server_obj
        self.rec_queue = rec_q
        self.server_obj.increase_reference_count()

    def __del__(self):
        print("delete server rc process:" + self.thread_name)
        self.server_obj.decrease_reference_count()

    def run(self):
        print("run server rc process:" + self.thread_name)
        while True:
            self.__rec_data()

    def get_rec_data(self):
        while not self.rec_queue.empty():
            data = self.rec_queue.get()
            if data is not None:
                print(data)
                return data

    def __rec_data(self):
        if self.server_obj.net_state:
            data = self.server_obj.client.recv(sn.data_size)
            if data:
                self.rec_queue.put(self.__pre_process(data))

    # 可以被子类重载
    def __pre_process(self, data):
        parameters = pickle.loads(data)
        return parameters


# 服务端的发送进程
class ServerSendBaseProcess(Process):

    def __init__(self, thread_name):
        super(ServerSendBaseProcess, self).__init__()
        self.thread_name = thread_name
        self.server_obj = None
        self.send_queue = None

    def init_para(self, server_obj, send_q):
        self.server_obj = server_obj
        self.send_queue = send_q
        self.server_obj.increase_reference_count()

    def __del__(self):
        print("delete server send process:" + self.thread_name)
        self.server_obj.decrease_reference_count()

    def run(self):
        print("run server send process:" + self.thread_name)
        while True:
            self.__send()

    def send_data(self, data):
        self.send_queue.put(data)

    def __send(self):
        if self.server_obj.net_state:
            if not self.send_queue.empty():
                data = self.send_queue.get()
                self.server_obj.client.send(self.__pre_process(data))

    # 可以被子类重载
    def __pre_process(self, data):
        return pickle.dumps(data)
