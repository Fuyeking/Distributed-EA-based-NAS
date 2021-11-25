import multiprocessing as mp
import socket

from dml import dml_server_process as dsp

data_size = 1024


class ServerProxy:

    def __init__(self, host, ip_port):
        '''
        :param host
        :param ip_port
        '''
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 用于通信的socket
        self.server_socket.bind((host, ip_port))
        self.net_state = False  # 网络连接状态
        self.__socket_reference_count = 0  # socket会在多个线程中被使用，引用次数，在析构过程中，引用次数为0，便可以直接删除
        self.send_process = None
        self.rec_process = None
        self.send_queue = mp.Queue()
        self.rec_queue = mp.Queue()
        self.client = None

    def __del__(self):
        self.__close_socket()

    def __close_socket(self):
        if self.__socket_reference_count == 0:
            self.server_socket.close()

    def increase_reference_count(self):
        self.__socket_reference_count += 1

    def decrease_reference_count(self):
        self.__socket_reference_count -= 1

    def create_conn(self):
        self.server_socket.listen(5)
        while not self.net_state:
            self.client, addr = self.server_socket.accept()
            print('client address:', addr)
            self.net_state = True

    def send_data(self, data):
        self.send_queue.put(data)

    def get_rec_data(self):
        if not self.rec_queue.empty():
            data = self.rec_queue.get()
            if data is not None:
                return data

    def start_send_request(self):
        self.client.send("OK".encode('utf-8'))

    def init_process(self):
        self.rec_process = dsp.ServerRecBaseProcess("server")
        self.rec_process.init_para(self, self.rec_queue)
        self.send_process = dsp.ServerSendBaseProcess("server")
        self.send_process.init_para(self, self.send_queue)

    def run_process(self):
        self.send_process.start()
        self.rec_process.start()

    def stop_process(self):
        self.send_process.terminate()
        self.rec_process.terminate()
