#include <stdio.h>

template<class T>
void swap(T & u, T & v) {
  T w = u;
  u = v;
  v = w;
}

int point(int i, int j, int k, int N) {
  return N*N*i+N*j+k;
}

int main() {
  //INPUTS
  float	L = 1.0; //1 cm size of the size of the box
  int N = 32; // intervals to divice L
  float h = L/N; // = 1/32 cm
    
  float *ha = new float[N*N*N];
  float *ha_tmp = new float[N*N*N];  
  float *hb = new float[N*N*N];
  float *da, *db;
  // cudaMalloc((void **)&da, N*N*N*sizeof(int));

  // set problem input (b)
  for (int i = 0; i<N; ++i) {
    for (int j = 0; j<N; ++j) {
      for (int k = 0; k<N; ++k) {
	hb[N*N*i+N*j+k] = 0.0;
      }
    }
  }
  hb[point(3,5,8,N)] = -1;
  hb[point(8,15,9,N)] = +2;
  hb[point(10,3,22,N)] = -1;
  
  // set initial conditions for solver
  for (int i = 0; i<N; ++i) {
    for (int j = 0; j<N; ++j) {
      for (int k = 0; k<N; ++k) {
	ha[N*N*i+N*j+k] = 0.0;
      }
    }
  }

  // boundary conditions a[p] is zero at box boundary
  for(int step=0; step<100; step++) { 
    for (int i = 1; i<N-1; ++i) {
      for (int j = 1; j<N-1; ++j) {
	for (int k = 1; k<N-1; ++k) {
	  int p = point(i,j,k,N);
	  int p_up = point(i+1,j,k,N);
	  int p_down = point(i-1,j,k,N);
	  int p_left = point(i,j-1,k,N);
	  int p_right = point(i,j+1,k,N);
	  int p_front = point(i,j,k-1,N);
	  int p_back = point(i,j,k+1,N);
	  ha_tmp[p] = 1.0/6.0*(h*h*hb[p] +
			      ha[p_up] + ha[p_down] +
			      ha[p_left] + ha[p_right] +
			      ha[p_front] + ha[p_back]);
	}
      }
    }
    swap(ha, ha_tmp);
  }

  // save ha contains result

  /*
  cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);
    add<<<N, 1>>>(da, db);
    cudaMemcpy(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i<N; ++i) {
        printf("%d\n", hb[i]);
    }
    cudaFree(da);
    cudaFree(db);
  */
  return 0;
}
