#include <mpi.h>
#include <iostream>
#include <vector>


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
  vector<int> v_init(N);  
  vector<int> v_final(N);  
  vector<int> v(M+2);
  vector<int> w(M+2);
  if (rank==0) {
    for(size_t k=0; k<N; k++) v_init[k] = k*k % 2;
  }

  v[0] = 0;
  v[M+1] = 0;
  check_error(MPI_Scatter(&v_init[0], M, MPI_INT, &v[1], M, MPI_INT, 0, MPI_COMM_WORLD));

  cout << rank << ":";
  for(int k=0; k<M+2; k++) cout << v[k] << ",";
  cout << endl;
  
  for(int step=0; step<3; step++) {
    // copy those boundaries to keep them in sync
    // cout << rank << ":sending\n";
    if(rank>0) {
      MPI_Isend(&v[1],1,MPI_INT,rank-1,0,MPI_COMM_WORLD,&rl);
    }
    if(rank<p-1) {
      MPI_Isend(&v[M],1,MPI_INT,rank+1,0,MPI_COMM_WORLD,&rr);
    }
    // cout << rank << ":receiving\n";
    if(rank>0) {
      MPI_Recv(&v[0],1,MPI_INT,rank-1,0,MPI_COMM_WORLD,&status_left);
    }
    if(rank<p-1) {
      MPI_Recv(&v[M+1],1,MPI_INT,rank+1,0,MPI_COMM_WORLD,&status_right);
    }
    // cout << rank << ":waiting\n";
    if(rank>0) {
      MPI_Wait(&rl, &srl);
    }
    if(rank<p-1) {
      MPI_Wait(&rr, &srr);
    }

    for(int k=1; k<M+1; k++) {
      if(v[k-1]==0 && v[k]==0 && v[k+1]==0) w[k]=0;
      else if(v[k-1]==0 && v[k]==1 && v[k+1]==0) w[k]=0;
      else if(v[k-1]==1 && v[k]==1 && v[k+1]==0) w[k]=1;
      else if(v[k-1]==0 && v[k]==1 && v[k+1]==1) w[k]=1;
      else if(v[k-1]==1 && v[k]==0 && v[k+1]==1) w[k]=1;
      else if(v[k-1]==1 && v[k]==1 && v[k+1]==1) w[k]=0;
    }
    v = w;
  
    //cout << rank << ":";
    //for(int k=0; k<M+2; k++) cout << v[k] << ",";
    //cout << endl;

    check_error(MPI_Gather(&v[1], M, MPI_INT, &v_final[0], M, MPI_INT, 0, MPI_COMM_WORLD));
    
    if (rank==0) {
      cout << "result:";
      for(int k=0; k<N; k++) cout << v_final[k] << ",";
      cout << endl;
    }
  }

  check_error(MPI_Finalize());

  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}
