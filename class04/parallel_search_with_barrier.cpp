#include <vector>
#include <iostream>
#include <thread>
#include <barrier>
#include <mutex>
#include "math.h"
#include "time.h"

using namespace std;

mutex mymutex;
barrier mybarrier(3);
			
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

void worker(const vector<int>& data, int x, size_t p, size_t rank, vector<int> & local_results) {
  size_t N = data.size();  
  size_t size = (N / p) + ((rank < N % p)?1:0);
  size_t index_start = rank * (N / p) + min(rank, N % p);
  size_t index_end = index_start + size;
  cout << "rank " << rank << " index_start " << index_start << endl;
  
  for(int i=index_start; i<index_end; i++) {
    if (data[i] == x) {
      local_results.push_back(i);      
    }
  }

  
  mybarrier.arrive_and_wait();

  sleep(3);
  
  cout << "rank " << rank << " done\n";
}


vector<int> parallel_search(const vector<int>& data, int x, int p) {
  vector<int> results;
  vector<thread*> threads;
  vector<vector<int>> local_results(p);
  for(size_t rank=0; rank<p; rank++) {
    threads.push_back(new thread([&,rank](){
				   worker(data, x, p, rank, local_results[rank]);
				 }));
  }
  std::cout << "computing\n";

  mybarrier.arrive_and_wait();
  
  for(size_t rank=0; rank<p; rank++) {
    for(auto element : local_results[rank]) {
      results.push_back(element);
    }
  }

  
  
  for(size_t rank=0; rank<p; rank++) {
    thread& t = *threads[rank];
    t.join();
    delete threads[rank];
  }
  threads.resize(0);

  return results;
}


int main(int argc, char** argv) {

  int N = 100;
  int p = 2;
  vector<int> data(N);
  int x;
  vector<int> results;

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
