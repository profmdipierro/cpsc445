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

__device__ int idx(int i, int j, int k, int N) {
  return N*N*i+N*j+k;
}

__global__ void solve(float * da, float * db, float *da_tmp, int N, float h) {
  for (int i = 1; i<N-1; ++i) {
    for (int j = 1; j<N-1; ++j) {
      for (int k = 1; k<N-1; ++k) {
	int p = idx(i,j,k,N);
	int p_up = idx(i+1,j,k,N);
	int p_down = idx(i-1,j,k,N);
	int p_left = idx(i,j-1,k,N);
	int p_right = idx(i,j+1,k,N);
	int p_front = idx(i,j,k-1,N);
	int p_back = idx(i,j,k+1,N);
	da_tmp[p] = 1.0/6.0*(h*h*db[p] +
			     da[p_up] + da[p_down] +
			     da[p_left] + da[p_right] +
			     da[p_front] + da[p_back]);
      }
    }
  }
}

int main() {
  //INPUTS
  float	L = 1.0; //1 cm size of the size of the box
  int N = 32; // intervals to divice L
  float h = L/N; // = 1/32 cm
    
  float *ha = new float[N*N*N];
  float *hb = new float[N*N*N];
  float *da, *db, *da_tmp;
  cudaMalloc((void **)&da, N*N*N*sizeof(int));
  cudaMalloc((void **)&db, N*N*N*sizeof(int));
  cudaMalloc((void **)&da_tmp, N*N*N*sizeof(int));

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

  cudaMemcpy(da, ha, N*N*N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb, N*N*N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(da_tmp, ha, N*N*N*sizeof(int), cudaMemcpyHostToDevice);
  // boundary conditions a[p] is zero at box boundary
  for(int step=0; step<100; step++) { // 100 even is important
    solve<<<1,1>>>(da, db, da_tmp, N, h);
    swap(da, da_tmp);
  }

  cudaMemcpy(hb, db, N*N*N*sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(da);
  cudaFree(db);
  cudaFree(da_tmp);
  free(ha);
  free(hb);
  return 0;
}
