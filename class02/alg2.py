import sys
import time
import socket
import threading
import random
import signal


class Pipe(threading.Thread):
    def __init__(self, conn):
        threading.Thread.__init__(self)
        self.conn = conn
        self.queue = None
        
    def run(self):
        while True:            
            message = self.conn.recv(1000)
            if message:
                self.queue = message
            time.sleep(0.0001)

    def recv(self):
        print(sys.argv[1], self.queue)
        while not self.queue:
            time.sleep(0.0001)
        print(sys.argv[1], "recv ...")
        message, self.queue = self.queue, None
        return message

def create_server(rank, p, port):
    pipes = {}
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("binding to", port)
    sock.bind(("127.0.0.1", port))
    sock.listen(100)
    for k in range(p):
        if k != rank:
            conn, address = sock.accept()
            pipe = Pipe(conn)
            pipe.start()
            pipes[k] = pipe
    return pipes

class Server(threading.Thread):
    def __init__(self, rank, p, port):
        threading.Thread.__init__(self)
        self.rank = rank
        self.p = p
        self.port = port
        self.senders = {}
        self.receivers = {}

    def run(self):
        self.receivers = create_server(self.rank, self.p, self.port + self.rank)

    def connect_clients(self):
        time.sleep(3)
        for k in range(self.p):
            if k != self.rank:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(("127.0.0.1", self.port + k))
                self.senders[k] = sock
                print(self.rank, "connected to", k)
        time.sleep(3)
        
def main():
    rank = int(sys.argv[1])
    p = int(sys.argv[2])
    print(f"{rank}/{p}")

    server = Server(rank, p, 9000)
    server.start()
    server.connect_clients()

    print(rank, "ready!")
    message = b"hello"

    if rank == 0:
        for k in range(1,p):
            print(rank,"sending to", k)
            server.senders[k].send(message)
            print(rank,"receiving from", k)
            message = server.receivers[k].recv()
            print(rank, "received from", k, message)
    else:
        print(rank,"receiving from", 0)
        message = server.receivers[0].recv()
        print(rank, "received", message)
        print(rank,"sending to", 0)
        server.senders[0].send(message)

    time.sleep(10)
    print(rank, "done")
                       
if __name__ == "__main__":
    main()
