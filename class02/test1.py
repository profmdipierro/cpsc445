import os
import sys
import argparse

def main():
    print(sys.argv)
    """
    for k in range(10):
        pid = os.fork()
        if pid == 0:
            print("exiting")
            break
    print(k)
    """

if __name__ == "__main__":
    main()
