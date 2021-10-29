#include <mpi.h>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <thread>
#include "math.h"
#include "unistd.h"

using namespace std;

void check_error(int status, const string message="MPI error") {
  if ( status != 0 ) {    
    cerr << "Error: " << message << endl;
    exit(1);
  }
}

void print(vector<int> & data, size_t index_start=0, size_t index_end=-1) {
  cout << "[";
  if (index_end == -1) index_end = data.size();
  for(int k=index_start; k<index_end; k++) {
    cout << data[k] << ((k < index_end-1)?",":"");
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
  mergesort(data, index_start, index_middle);
  mergesort(data, index_middle, index_end);
  merge(data, index_start, index_middle, index_end);
}


int main(int argc, char** argv) {

  int p, rank;
  
  // Initialized MPI
  check_error(MPI_Init(&argc, &argv), "unable to initialize MPI");
  check_error(MPI_Comm_size(MPI_COMM_WORLD, &p), "unable to obtain p");
  check_error(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "unable to obtain rank");
  cout << "Starting process " << rank << "/" << "p\n";

  int N = 4 * p;
  int niter = (int) (log(p)/log(2) + 0.0001);
  vector<int> data(N);

  // if rank is zero we initialize
  if (rank == 0) {
    for(int i=0; i<N; i++) {
      data[i] = rand() % 1000;
    }
    print(data, 0, data.size());
  }

  int items_node = N / p;
  MPI_Status status;

  for(int k=1; k<niter+1; k++) {
    int shift = 0x1 << (niter - k);
    int block = shift * items_node;
    // sleep(rank);
    if (rank % shift == 0) {
      int pos = rank / shift;
      if(pos % 2 == 0) {
	int dest = rank + shift;
	// cout << "step" << k << " " << rank << " -> " << dest << endl;
	MPI_Send(&data[block], block, MPI_INT, dest, 0, MPI_COMM_WORLD);
      } else if(pos % 2 == 1) {
	int src = rank - shift;
	// cout << "step" << k << " " << rank << " <- " << src << endl;
	MPI_Recv(&data[0], block, MPI_INT, src, 0, MPI_COMM_WORLD, &status);
      }
    }
  }

  mergesort(data, 0, items_node);

  for(int k=niter; k>0; k--) {
    int shift = 0x1 << (niter - k); 
    int block = shift * items_node;
    if (rank % shift == 0) {
      int pos = rank / shift;
      if(pos % 2 == 0) {
	int src = rank + shift;
	MPI_Recv(&data[block], block, MPI_INT, src, 0, MPI_COMM_WORLD, &status);
	merge(data, 0, block, 2*block);
	// print(data, 0, 2*block);
      } else if(pos % 2 == 1) {
	int dest = rank - shift;
	MPI_Send(&data[0], block, MPI_INT, dest, 0, MPI_COMM_WORLD);
      }
    }
    shift *=2;
  }

  // if rank is zero we print
  if (rank == 0) {
    print(data, 0, data.size());
  }
     
  check_error(MPI_Finalize());

  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}
