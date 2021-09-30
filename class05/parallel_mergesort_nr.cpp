#include <vector>
#include <iostream>
#include <stdlib.h>
#include <thread>

using namespace std;

void print(vector<int> & data, size_t index_start, size_t index_end) {
  cout << "[";
  for(int k=index_start; k<index_end; k++) {
    cout << data[k] << ",";
  }
  cout << "]" << endl;
}


void merge(vector<int> & data, size_t index_start, size_t index_middle, size_t index_end) {
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

void mergesort_nr(vector<int> & data) {
  int blocksize = 1;
  int n = data.size();
  int j,k;  
  vector<std::thread*> threads;
  while (blocksize < n) {
    std::cout << "blocksize=" << blocksize << std::endl;
    for (int i=0; i<n; i+=2*blocksize) {
      j = i + blocksize;
      k = std::min(n, j + blocksize);
      if (k > i) {
	// do this in  thread
	std::cout << "i,j,k=" << i << ", " << j << "," << k << std::endl;
	threads.push_back(new std::thread([&,i,j,k](){ merge(data, i,j,k); }));	
      }
    }

    for (int i=0; i<threads.size(); i++) {
      threads[i]->join();
    }
    threads.resize(0);
    std::cout << "joined\n";
    print(data, 0, n);
    blocksize *= 2;
  }
}

int main(int argc, char** argv) {

  int N = 10;
  // int p = 2;
  vector<int> data(N);

  for(int i=0; i<N; i++) {
    data[i] = rand() % 1000;
  }
     
  mergesort_nr(data);

  print(data, 0, data.size());
  
  return 0;
}
