import time
import threading

memory = {"B": 2, "C": 5, "D": 9, "Z": 0}

def g_memory(m):
    m["Z"] = g(m["C"], m["D"])

def f(x,y):
    time.sleep(1)
    return x + y

def g(x,y):
    time.sleep(1)
    return x * y

def main():
    a,b,c,d = 0,2,5,9
    t = threading.Thread(target=lambda m=memory: g_memory(m))
    t.start()
    z = f(c,d)
    t.join()
    u = f(z, memory["Z"])
    print(u)

if __name__ == "__main__":
    main()
