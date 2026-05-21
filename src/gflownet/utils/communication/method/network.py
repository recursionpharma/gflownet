import pickle
import socket
import struct
from typing import Any

from gflownet.utils.communication.method.base import IPCModule

PACKET_SIZE = 4096


class NetworkIPC(IPCModule):
    """IPCModule using TCP/IP Socket Networking"""

    def __init__(
        self,
        process_type: str,
        host: str = "localhost",
        port: int = 1428,
    ):
        """Initialization of IPC Module

        Parameters
        ----------
        process_type : str
            - GFNTask: 'sampler'
            - RewardModule: 'reward'
        host: str
            host for networking
        port: int
            port for networking

        """
        super().__init__(process_type)
        self.host = host
        self.port = port

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.is_connected = False  # conneciton btw gfn sampler and reward process
        if self.process_type == "sampler":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((host, port))
            self.socket.listen(1)

        self.received_data: bytes | None = None
        self.data_length: int | None = None
        self.is_terminate: bool = False

    ### GFN ###
    def sampler_from_reward(self) -> tuple[list[list[float]], list[bool]]:
        self.assert_is_sampler()
        assert self.received_data is not None
        obj_props, is_valid = pickle.loads(self.received_data)  # unserialized
        self.received_data = None
        self.data_length = None
        return obj_props, is_valid

    def sampler_to_reward(self, objs: list[Any]) -> bool:
        self.assert_is_sampler()
        request_data = pickle.dumps(objs)
        request_size = struct.pack("!I", len(request_data))

        if not self.is_connected:
            self.conn, self.addr = self.socket.accept()  # accept connection btw gfn - reward process
            self.conn.settimeout(10)
            self.is_connected = True
        self.conn.send(request_size)
        self.conn.sendall(request_data)
        return True

    def sampler_wait_reward(self) -> bool:
        self.assert_is_sampler()
        # collect data packets
        # TODO: prevent packet error - if damaged, repeat
        try:
            # data length first to prevent data loss
            if self.data_length is None:
                assert self.received_data is None
                data_size_byte = self.conn.recv(4)
                assert data_size_byte, "connection is broken"
                self.data_length = struct.unpack("!I", data_size_byte)[0]
                self.received_data = b""

            if self.received_data is not None:
                assert self.data_length is not None
                while len(self.received_data) < self.data_length:
                    chunk = self.conn.recv(min(PACKET_SIZE, self.data_length - len(self.received_data)))
                    assert chunk, "connection is broken"
                    self.received_data += chunk
                assert len(self.received_data) == self.data_length
                return False

        # if packet is not receied for 10 seconds, return to wait cycle
        except socket.timeout:
            return True
        return True

    def sampler_unlock_reward(self):
        self.assert_is_sampler()

    def sampler_terminate_reward(self):
        self.assert_is_sampler()
        self.conn.send(b"")
        self.conn.close()
        self.socket.close()

    ### Reward Module ###
    def reward_from_sampler(self) -> list[Any]:
        self.assert_is_reward()
        assert self.received_data is not None
        objs = pickle.loads(self.received_data)  # unserialized
        self.data_length = None
        self.received_data = None
        return objs

    def reward_to_sampler(self, obj_props: list[list[float]], is_valid: list[bool]):
        self.assert_is_reward()
        response_data = pickle.dumps((obj_props, is_valid))
        response_size = struct.pack("!I", len(response_data))
        self.socket.send(response_size)
        self.socket.sendall(response_data)

    def reward_wait_sampler(self) -> bool:
        self.assert_is_reward()
        # if reward process is not connected, wait gfn sampler.
        if not self.is_connected:
            try:
                self.socket.connect((self.host, self.port))
                self.socket.settimeout(10)
                self.is_connected = True
            except ConnectionRefusedError:
                return True

        # collect data packets
        # TODO: prevent packet error - if damaged, repeat
        try:
            # data length first
            if self.data_length is None:
                assert self.received_data is None
                data_size_byte = self.socket.recv(4)
                if not data_size_byte:
                    # when sampler is terminated, it sends empty packet.
                    self.is_terminate = True
                else:
                    self.data_length = struct.unpack("!I", data_size_byte)[0]
                    self.received_data = b""

            if self.received_data is not None:
                assert self.data_length is not None
                while len(self.received_data) < self.data_length:
                    chunk = self.socket.recv(min(PACKET_SIZE, self.data_length - len(self.received_data)))
                    assert chunk, "connection is broken"
                    self.received_data += chunk
                assert len(self.received_data) == self.data_length
                return False

        # if packet is not receied for 10 seconds, return to wait cycle
        except socket.timeout:
            return True
        return True

    def reward_unlock_sampler(self):
        self.assert_is_reward()

    def reward_is_terminated(self):
        self.assert_is_reward()
        return self.is_terminate
