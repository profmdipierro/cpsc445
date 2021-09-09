Import copy
import sys
import threading
import time

class Pipe:
    def __init__(self):
        self.data = None
        self.semaphore = threading.Event()
    def send(self, message):
        message = copy.deepcopy(message)
        self.data = message
        self.semaphore.set()
    def recv(self):
        self.semaphore.wait()
        item = self.data
        self.semaphore.clear()
        return item

class PingPong(threading.Thread):

    def __init__(self, n=1000):
        threading.Thread.__init__(self)
        self.n = n
        self.pipe1 = Pipe()
        self.pipe2 = Pipe()

    def run(self):
        for k in range(self.n):
            message = self.pipe1.recv()
            # print("second thread received", message)
            # message = message + 1
            self.pipe2.send(message)
            

    def main(self):
        message = ['x']*int(sys.argv[1])
        self.start()
        t0 = time.time()
        for k in range(self.n):
            self.pipe1.send(message)
            # print("main sent", message)
            message = self.pipe2.recv()
        t1 = time.time()
        self.join()        
        print((t1-t0)/self.n/2)
    
def main():
    m = PingPong()
    m.main()

if __name__ == "__main__":
    main()
