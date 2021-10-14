#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <vector>
#include "math.h"
#include "assert.h"

using namespace std;

// ./a.out 10
class MyMPI {
public:
  size_t rank;
  size_t p;
  vector<int> fds;
  string network_topology;
  MyMPI(int argc, char**argv) {
    this->network_topology = argv[2];
    assert(argc > 1);
    p = atoi(argv[1]);
    rank = 0;
    cout << p << endl;

    fds.resize(2*p*p);
    for(size_t i=0; i<p; i++) {
      for(size_t j=0; j<p; j++) {
	// send data from i to j
	pipe(&fds[2*(i*p+j)]);
      }
    }

    for(size_t k=1; k<p; k++) {
      if(fork() == 0) {
	rank = k;
	break;
      }
    }
  }
  bool check_connection(int i, int j) {
    if (network_topology == "switch") return true;
    if (network_topology == "bus") return true;
    if (network_topology == "1d-mesh") return fabs(i-j) <= 1;
    if (network_topology == "1d-ring") return fabs((i - j + p) % p) <= 1;
    if (network_topology == "2d-mesh") {
      int s = (int) sqrt(p);
      int xi = (int) i/s, yi = i % s, xj = (int) j / s, yj = j % s;	
      return (xi-xj)*(xi-xj) + (yi-yj)*(yi-yj) <= 1;
    }
    if (network_topology == "tree") {
      return (i == 2*j +1 || i == 2*j +2 || j == 2*i+1 || j == 2*i + 2);
    }
    if (network_topology == "2d-ring") {
    }
    if (network_topology == "hypercube") {
    }
    return false;
  }
  int get_read_fd(int i) {
    if (!check_connection(rank, i)) throw runtime_error("Not connected");    
    return fds[2*(i*p+rank)+0];
  }
  int get_write_fd(int j) {
    if (!check_connection(rank, j)) throw runtime_error("Not connected");
    return fds[2*(rank*p+j)+1];
  }
  template<class T>
  void send(int j, const T & data) {
    write(get_write_fd(j), &data, sizeof(data));
  }
  template<class T>
  T recv(int i) {
    T data;
    read(get_read_fd(i), &data, sizeof(data));
    return data;
  }
  template<class T>
  void send(int j, const T * data, size_t size) {
    write(get_write_fd(j), data, sizeof(data) * size);
  }
  template<class T>
  void recv(int i, T * data, size_t size) {
    read(get_read_fd(i), data, sizeof(data)*size);
  }
  template<class T>
  void send_vec(int j, const vector<T> & data) {
    size_t size = data.size();
    send(j, size);
    send(j, &data[0], size);
  }
  template<class T>
  vector<T> recv_vec(int i) {
    size_t size = recv<size_t>(i);
    vector<T> data(size);
    recv(i, &data[0], size);
    return data;
  }
  template<class T>
  vector<T> scatter(const vector<T> & data, int source) {
    if(rank == source) {
      for(size_t destination=0; destination<p; destination++) {
	size_t N = data.size();
	size_t size = (N / p) + ((destination < N % p)?1:0);
	size_t index_start = destination * (N / p) + min(destination, N % p);
	send(destination, size);
	send(destination, &data[index_start], size);
      }
    }
    return recv_vec<T>(source);
  }
  template<class T>
  T broadcast_switch(const T& obj, int source) {    
    if (rank == source) {
      for(size_t dest=0; dest<p; dest++) {
	send(dest, obj);
      }
    }
    return recv<T>(source);
  }
  template<class T>
  T broadcast_tree(const T& obj, int source) {
    T tmp = obj;
    if (source != 0) { 
      int child=-1, k = source;
      while (k>0) {
	if(rank == 0) cout << "k=" << k << " and child=" << child << endl;
	int parent = (k - 1) / 2;	
	if (k == rank) {	  
	  if (rank != source && rank!=child) {
	    cout << rank << " <- " << child << endl;
	    tmp = recv<T>(child);
	  }	  
	  if (rank != 0 && rank != parent) {
	    cout << rank << " -> " << parent << endl;
	    send(parent, tmp);
	  }	  
	}
	child = k;
	k = parent;
      }
      if (source != 0 && rank == 0) {
	tmp = recv<T>(child);
      }
    }
    if (rank != 0) {
      int parent = (rank - 1)/2;
      tmp = recv<T>(parent);
    }
    int i = 2*rank + 1, j = 2*rank + 2;
    if (i < p) send(i, tmp);
    if (j < p) send(j, tmp);
    return tmp;
  }

  template<class T>
  T broadcast(const T& obj, int source) {
    if (network_topology=="switch" || network_topology=="bus") return broadcast_switch(obj, source);
    if (network_topology=="tree") return broadcast_tree(obj, source);
    throw runtime_error("Network not supported");
  }
};

int main(int argc, char ** argv) {

  MyMPI comm(argc, argv);
  /* MyMPI comm("switch"); */

  cout << "my rank is " << comm.rank << endl; 
  cout << "p " << comm.p << endl;

  int x = 0;
  int bcaster = 3;
  if(comm.rank == bcaster) {
    x = 456346;
  }
  x = comm.broadcast(x, bcaster);

  cout << "x = " << x << endl;

  /*
  comm.send(data, 2);
  data = comm.recv(3);
  v = comm.scatter(data);
  data = comm.collect(v, 0);
  comm.barrier();
  sum_of_cs = comm.reduce_sum(c);
  */
  return 0;
}
