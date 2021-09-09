import socket
import threading

class Pipe(threading.Thread):
    def __init__(self, conn):
        threading.Thread.__init__(self)
        self.conn = conn
        self.queue = []
        
    def run(self):
        while True:
            message = self.conn.recv(1000)
            if not message:
                break
            print("I received", message)
            self.queue.append(message)

def main(port=9001):
    threads = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1",port))
    sock.listen(100)
    while True:
        conn, address = sock.accept()
        thread = Pipe(conn)
        thread.start()
        threads.append(thread)
        
if __name__ == "__main__":
    main()
