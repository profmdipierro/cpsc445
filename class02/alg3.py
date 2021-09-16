import os
import sys
import time
import socket
import threading
import random
import signal


class Pipe:
    def __init__(self, conn):
        self.conn = conn

    def send(self, message):
        self.conn.send(message.encode())
        self.conn.send(b"\x00")

    def recv(self):
        message = b""
        while True:
            c = self.conn.recv(1)
            if c == b"\x00":
                return message.decode()
            message += c

class Server(threading.Thread):
    def __init__(self, rank, p, port):
        threading.Thread.__init__(self)
        self.rank = rank
        self.p = p
        self.port = port
        self.senders = {}
        self.receivers = {}

    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("binding to", self.port + self.rank)
        self.sock.bind(("127.0.0.1", self.port + self.rank))
        self.sock.listen(self.p - 1)
        while len(self.receivers) < self.p - 1:
            conn, address = self.sock.accept()
            pipe = Pipe(conn)
            k = int(pipe.recv())
            self.receivers[k] = pipe
        print("server stopped")

    def connect_clients(self):
        time.sleep(3)
        for k in range(self.p):
            if k != self.rank:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(("127.0.0.1", self.port + k))
                self.senders[k] = Pipe(sock)
                self.senders[k].send(str(self.rank))
                print(self.rank, "connected to", k)


def main():
    rank = int(sys.argv[1])
    p = int(sys.argv[2])
    print(f"{rank}/{p}")

    server = Server(rank, p, 9100)
    server.start()
    server.connect_clients()
    server.join()

    print(rank, "ready!")
    message = "hello"

    if rank == 0:
        print(rank, "sending to", 1)
        server.senders[1].send(message)
        print(rank, "receiving from", p - 1)
        message = server.receivers[p - 1].recv()
        print(rank, "received from", p - 1, message)
    else:
        previous = rank - 1
        print(rank, "receiving from", previous)
        message = server.receivers[previous].recv()
        print(rank, "received from", previous, message)
        message = message + "."
        next = (rank + 1) % p
        print(rank, "sending to", next, message)
        server.senders[next].send(message)

    time.sleep(10)
    print(rank, "done")


if __name__ == "__main__":
    main()
