#include <mpi.h>
#include <iostream>
#include <vector>
#include "math.h"

using namespace std;

void check_error(int status, const string message="MPI error") {
  if ( status != 0 ) {    
    cerr << "Error: " << message << endl;
    exit(1);
  }
}

int main (int argc, char *argv[]) {
  int rank;
  int p;

  // Initialized MPI
  check_error(MPI_Init(&argc, &argv), "unable to initialize MPI");
  check_error(MPI_Comm_size(MPI_COMM_WORLD, &p), "unable to obtain p");
  check_error(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "unable to obtain rank");
  cout << "Starting process " << rank << "/" << "p\n";

  MPI_Status status_left, status_right, srl, srr;
  MPI_Request rl, rr;
  
  int M = 5;
  int N = p*M;
  float domain = 3; // meters
  float h = domain / N;
  float sum_local, sum_global;

  vector<float> f_init(N);  
  vector<float> f_final(N);  
  vector<float> f(M+2);
  vector<float> w(M+2);

  vector<float> g_global(N);  
  vector<float> g_local(N);  


  if (rank==0) {
    for(int i=0; i<N; i++) g_global[i]=3*sin(h*i);
    for(int i=0; i<N; i++) f[i]=0;
    // set inputs
  }

  f[0] = 0; // INPUT
  f[M+1] = 0;
  check_error(MPI_Scatter(&f_init[0], M, MPI_FLOAT, &f[1], M, MPI_FLOAT, 0, MPI_COMM_WORLD));

  check_error(MPI_Scatter(&g_global[0], M, MPI_FLOAT, &g_local[1], M, MPI_FLOAT, 0, MPI_COMM_WORLD));

  cout << rank << ":";
  for(int k=0; k<M+2; k++) cout << f[k] << ",";
  cout << endl;
  
  for(int step=0; step<10000; step++) {
    // copy those boundaries to keep them in sync
    // cout << rank << ":sending\n";
    if(rank>0) {
      MPI_Isend(&f[1],1,MPI_FLOAT,rank-1,0,MPI_COMM_WORLD,&rl);
    }
    if(rank<p-1) {
      MPI_Isend(&f[M],1,MPI_FLOAT,rank+1,0,MPI_COMM_WORLD,&rr);
    }
    // cout << rank << ":receiving\n";
    if(rank>0) {
      MPI_Recv(&f[0],1,MPI_FLOAT,rank-1,0,MPI_COMM_WORLD,&status_left);
    }
    if(rank<p-1) {
      MPI_Recv(&f[M+1],1,MPI_FLOAT,rank+1,0,MPI_COMM_WORLD,&status_right);
    }
    // cout << rank << ":waiting\n";
    if(rank>0) {
      MPI_Wait(&rl, &srl);
    }
    if(rank<p-1) {
      MPI_Wait(&rr, &srr);
    }

    for(int i=1; i<M+1; i++) {
      w[i] = -1.0/2.0*(h*h*g_local[i]  - f[i-1] - f[i+1]);
    }

    sum_local = 0.0;
    for(int i=1; i<M+1; i++) {
      sum_local += pow(f[i] - w[i], 2);
    }
    cout << rank << " sum_local " << sum_local << endl;
    sum_global = 0;

    MPI_Allreduce(&sum_local, &sum_global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    f = w;

    cout << rank << " here\n";
  
    //cout << rank << ":";
    //for(int k=0; k<M+2; k++) cout << v[k] << ",";
    //cout << endl;

    check_error(MPI_Gather(&f[1], M, MPI_FLOAT, &f_final[0], M, MPI_FLOAT, 0, MPI_COMM_WORLD));
    
    if (rank==0) {
      if(step > 0) cout << "step " << step << " sum_global " << sum_global << endl;
      cout << "result:";
      for(int k=0; k<N; k++) cout << f_final[k] << ",";
      cout << endl;
    }
    if (step > 0 && sum_global < 0.01) break;
  }

  check_error(MPI_Finalize());

  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}
