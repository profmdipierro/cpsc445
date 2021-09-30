#include <vector>
#include <iostream>
#include <stdlib.h>
#include <thread>
#include <queue>
#include "unistd.h"

using namespace std;

void print(vector<int> & data, size_t index_start, size_t index_end) {
  cout << "[";
  for(int k=index_start; k<index_end; k++) {
    cout << data[k] << ",";
  }
  cout << "]" << endl;
}


void merge(vector<int> & data, size_t index_start, size_t index_middle, size_t index_end) {
  cout << "merging\n";
  int k=0, i = index_start, j=index_middle;
  vector<int> copy(index_end - index_start);
  
  while(k<copy.size()){
    if (i < index_middle && j < index_end)
      if (data[i] < data[j]) {
	copy[k++] = data[i++];
      } else {
	copy[k++] = data[j++];
      }
    else if (i < index_middle) {
      copy[k++] = data[i++];
    }
    else {
      copy[k++] = data[j++];
    }
  }
  for(int k=0; k<copy.size(); k++) {
    data[k + index_start] = copy[k];
  }
}

class Task {
public:
  vector<int> * data_ptr;
  int i,j,k;
  Task(vector<int> & data, int i, int j, int k) {
    this->data_ptr = &data;
    this->i = i;
    this->j = j;
    this->k = k;    
  }       
  void run() {
    merge(*data_ptr, i,j,k);
  }
};

class TaskQueue {
public:
  queue<Task> tasks;
  void push(Task task) {
    tasks.push(task);
    // tell the other method that there is something in the tasks queue
  }
  void task_runner() {
    // if there is a task in queue call task.run() and remove from queue
    while (tasks.size()) {
      tasks.front().run();
      tasks.pop();
    }
  }
};

void mergesort_nr(vector<int> & data, TaskQueue & task_queue) {
  int blocksize = 1;
  int n = data.size();
  int j,k;  
  // vector<std::thread*> threads;
  while (blocksize < n) {
    std::cout << "blocksize=" << blocksize << std::endl;
    for (int i=0; i<n; i+=2*blocksize) {
      j = i + blocksize;
      k = std::min(n, j + blocksize);
      if (k > i) {
	// do this in  thread
	std::cout << "i,j,k=" << i << ", " << j << "," << k << std::endl;
	task_queue.push(Task(data, i, j, k));
      }
    }
    blocksize *= 2;
  }
}

int main(int argc, char** argv) {

  int N = 10;
  vector<int> data(N);

  for(int i=0; i<N; i++) {
    data[i] = rand() % 1000;
  }

  TaskQueue task_queue;
  mergesort_nr(data, task_queue);
  std::thread mythread([&](){ task_queue.task_runner(); });

  mythread.join();

  print(data, 0, data.size());
  
  return 0;
}
