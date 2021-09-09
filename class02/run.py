#!/usr/bin/env python3.8
import os
import sys
import subprocess

def main():
    assert sys.argv[1] == '-p'
    p = int(sys.argv[2])
    command = sys.argv[3:]
    for k in range(1, p):
        pid = os.fork()
        if pid == 0:
            subprocess.check_call(command + [str(k), str(p)])
            return
    k = 0
    subprocess.check_call(command + [str(k), str(p)])

if __name__ == "__main__":
    main()
