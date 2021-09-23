#include <vector>
#include <iostream>
#include <stdlib.h>

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


void mergesort(vector<int> & data, size_t index_start, size_t index_end) {
  if (index_end - index_start < 2) return;
  size_t index_middle = (index_end + index_start) / 2;
  print(data, index_start, index_end);
  mergesort(data, index_start, index_middle);
  mergesort(data, index_middle, index_end);
  cout << "pre merge   ";
  print(data, index_start, index_end);
  merge(data, index_start, index_middle, index_end);
  cout << "after merge ";  
  print(data, index_start, index_end);
}

int main(int argc, char** argv) {

  int N = 10;
  // int p = 2;
  vector<int> data(N);

  for(int i=0; i<N; i++) {
    data[i] = rand() % 1000;
  }
     
  mergesort(data, 0, data.size());

  print(data, 0, data.size());
  
  return 0;
}
