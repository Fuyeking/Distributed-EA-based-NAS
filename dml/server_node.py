from dml import packet
from dml import server_proxy as sn

module = __import__("dml.dml_server_process")


class ServerNode:
    def __init__(self, ip_set, raw_data, raw_data_len):
        self.ip_set = ip_set
        self.server_nodes = {}
        self.rec_count = 0
        self.send_data_left = raw_data_len
        self.termination = False
        self.sharing_send_data = raw_data
        self.sharing_rec_data = []

    def distributed_dnn(self):
        self._create_server_nodes()  # 根据计算节点的个数创建对应的通信节点
        self._init_socket_conn()  # 和计算节点建立连接
        self._create_send_rec_process()  # 每个通信节点创建两个进程（负责收、发）
        self._start_send_rec_process()  # 开启进程
        self._notify_clients()  # 通知所有的计算节点可以开始发送数据

    def _create_server_nodes(self):
        for port, ip in self.ip_set.items():
            node = sn.ServerProxy(ip, port)
            self.server_nodes[port] = node

    def _init_socket_conn(self):
        for port, ip in self.ip_set.items():
            node = self.server_nodes[port]
            while not node.net_state:
                node.create_conn()

        # 允许被子类重载

    def _create_send_rec_process(self):
        '''
       create send and rec processes
        :return:
        '''
        for port, ip in self.ip_set.items():
            node = self.server_nodes[port]
            node.init_process()

    def _start_send_rec_process(self):
        for port, ip in self.ip_set.items():
            node = self.server_nodes[port]
            node.run_process()

    def _notify_clients(self):
        for port, ip in self.ip_set.items():
            node = self.server_nodes[port]
            node.start_send_request()

    def close_process(self):
        for port, ip in self.ip_set.items():
            node = self.server_nodes[port]
            node.stop_process()

    def get_rec_data(self, port):
        server_node = self.server_nodes[port]
        return server_node.get_rec_data()

    def add_send_data(self, port, data):
        server_node = self.server_nodes[port]
        server_node.send_data(data)

    def _create_packet(self, data_len):
        data = self.sharing_send_data[0:data_len]
        self.send_data_left = self.send_data_left - data_len
        self.sharing_send_data = self.sharing_send_data[data_len:]
        send_pk = packet.Packet(op_type=1, data_len=len(data), data_set=data, handle_len=0)
        send_pk.termination = self.termination
        return send_pk

    def controller(self):
        '''
        inheriting this class, different algorithms can be implemented independently
        '''
        return
