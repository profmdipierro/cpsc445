import time
import threading

#memory = {"B": 2, "C": 5, "D": 9}

class GThread(threading.Thread):

    def __init__(self, x, y):
        threading.Thread.__init__(self)
        self.x = x
        self.y = y

    def run(self):
        self.z = g(self.x, self.y)

def f(x,y):
    time.sleep(1)
    return x + y

def g(x,y):
    time.sleep(1)
    return x * y

def main():
    a,b,c,d = 0,2,5,9
    t = GThread(c,d)
    t.start()
    z = f(c,d)
    t.join()
    u = f(z, t.z)
    print(u)

if __name__ == "__main__":
    main()
