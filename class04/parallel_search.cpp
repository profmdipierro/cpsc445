#include <vector>
#include <set>
#include <iostream>
#include <thread>
#include <mutex>
#include "math.h"

using namespace std;

mutex mymutex;

/*
set<int> search(const vector<int>& data, int x, int offset) {
  set<int> results;
  for(int i=0; i<data.size(); i++) {
    if (data[i] == x) {
      results.insert(i + offset);
    }
  }
  return results;
}
*/

void worker(const vector<int>& data, int x, size_t p, size_t rank, set<int> & results) {
  size_t N = data.size();
  size_t size = (N / p) + ((rank < N % p)?1:0);
  size_t index_start = rank * (N / p) + min(rank, N % p);
  size_t index_end = index_start + size;

  cout << "rank " << rank << " index_start " << index_start << endl;
  
  for(int i=index_start; i<index_end; i++) {
    if (data[i] == x) {
      const lock_guard<mutex> lock(mymutex);
      results.insert(i);      
    }
  }
}


set<int> parallel_search(const vector<int>& data, int x, int p) {
  set<int> results;
  vector<thread*> threads;
  for(size_t rank=0; rank<p; rank++) {
    threads.push_back(new thread([&,rank](){ worker(data, x, p, rank, results); }));
  }
  std::cout << "computing\n";
  for(size_t rank=0; rank<p; rank++) {
    thread& t = *threads[rank];
    t.join();
    delete threads[rank];
  }
  threads.resize(0);
  return results;
}


int main(int argc, char** argv) {

  int N = 10;
  int p = 2;
  vector<int> data(N);
  int x;
  set<int> results;

  for(int i=0; i<N; i++) {
    data[i] = i  %  4;
  }
  x = 2;
    
  results = parallel_search(data, x, p);

  for(int element : results) {
    cout << element << endl;
  }
  
  return 0;
}
