import time
import threading

class MyClass(threading.Thread):

    def __init__(self, b, c, d):
        threading.Thread.__init__(self)
        self.a = 0
        self.b, self.c, self.d = b, c, d

    def run(self):
        self.z = g(self.c, self.d)

    def main(self):
        self.start()
        z = f(self.c,self.d)
        self.join()
        self.a = f(z, self.z)
        return self.a
    
def f(x,y):
    time.sleep(1)
    return x + y

def g(x,y):
    time.sleep(1)
    return x * y

def main():
    m = MyClass(b=2, c=5, d=9)
    a = m.main()
    print(a)

if __name__ == "__main__":
    main()
