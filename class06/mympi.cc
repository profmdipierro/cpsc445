#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <vector>
#include "assert.h"

using namespace std;

// ./a.out 10
class MyMPI {
public:
  size_t rank;
  size_t p;
  vector<int> fds;
  string network_topology;
  MyMPI(int argc, char**argv, string network_topology) {
    this->network_topology = network_topology;
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
  int get_read_fd(int i) {
    return fds[2*(i*p+rank)+0];
  }
  int get_write_fd(int j) {
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
};

int main(int argc, char ** argv) {

  MyMPI comm(argc, argv, "switch");
  /* MyMPI comm("switch"); */

  cout << "my rank is " << comm.rank << endl; 
  cout << "p " << comm.p << endl;

  vector<int> v;  
  if (comm.rank == 0) {
    /// pretent reading from a file
    v.resize(comm.p*3);
    for(int i=0; i<comm.p*3; i++) v[i] = i*i;
  }

  auto w = comm.scatter(v, 0);
  
  cout << "I am " << comm.rank << " and I got ";
  for(int j=0; j<w.size(); j++) cout << w[j] << " ";
  cout << endl;

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
