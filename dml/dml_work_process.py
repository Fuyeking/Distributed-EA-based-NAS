import pickle
from multiprocessing import Process

# import threading
from dml import server_proxy as sn


class WorkBaseSendProcess(Process):
    def __init__(self, name, send_client):
        super(WorkBaseSendProcess, self).__init__()
        self.name = name
        self.send_client = send_client
        self.send_client.increase_socket_reference_count()
        self.send_queue = send_client.send_queue
        print("create work send process:" + self.name)

    def __del__(self):
        print("delete work send process:" + self.name)
        self.send_client.decrease_socket_reference_count()

    def run(self):
        print("start work send process:" + self.name)
        while True:
            self.send()

    def send(self):
        if self.send_client.net_ready:
            if not self.send_queue.empty():
                data = self.send_queue.get()
                if data is not None:
                    self.send_client.server_socket.send(self.__pre_process(data))

    def __pre_process(self, data):
        return pickle.dumps(data)


class WorkBaseRecProcess(Process):

    def __init__(self, name, rec_client):
        super(WorkBaseRecProcess, self).__init__()
        self.name = name
        self.rec_client = rec_client
        self.rec_client.increase_socket_reference_count()
        self.rec_queue = rec_client.rec_queue
        print("create work rec thread:" + self.name)

    def __del__(self):
        print("delete work rec process:" + self.name)
        self.rec_client.decrease_socket_reference_count()

    def run(self):
        print("start work rec process:" + self.name)
        while True:
            self.__rec_data()

    def __rec_data(self):
        if self.rec_client.net_ready:
            data = self.rec_client.server_socket.recv(sn.data_size)
            if data:
                self.rec_queue.put(self.__pre_process(data))

    def __pre_process(self, data):

        '''
        子类继承后，根据应用场景不同，进行重载
        :param data:
        :return:
        '''
        return pickle.loads(data)
