#include <iostream>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <thread>

void run() {
  std::cout << "before\n";
  sleep(3);
  std::cout << "after\n";
}

int main(int argc, char** argv) {
  std::thread mythread(run);

  std::cout << "hello world\n";

  mythread.join();
  return 0;
}
