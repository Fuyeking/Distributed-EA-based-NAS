class Packet(object):

    def __init__(self, op_type=0, data_len=0, data_set=None, handle_len=0):
        self.op_type = op_type  # 0 represents applying for new data,1 represents returning processed data
        self.ind_len = data_len  # the number of individuals carried by a message
        self.ind_set = data_set  # store individuals (i.e., an individual represents a network)
        self.handle_len = handle_len  # the length of the processed data actually
        self.termination = False  # True means the end of evolution
