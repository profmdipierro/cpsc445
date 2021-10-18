#include <mpi.h>
#include <iostream>

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

  // example code
  int n = (rank==0?5:0), sum = 0;
  check_error(MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD));  
  check_error(MPI_Reduce(&sum, &n, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
  if (rank==0) {
    if (sum != n*p) { cerr << "error!\n"; exit(1); }
  }

  check_error(MPI_Finalize());

  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}
