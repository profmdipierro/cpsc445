#include <stdio.h>
#include <stdlib.h>

/*
__global__ void sort(int * da, int N) {
}
*/

template<class T>
void swap(T& a, T&b) {
  T c=a;
  a=b;
  b=c;
}

__global__ void sort(float * a, int N) {
  int i = threadIdx.x;
  for (int k = 2; k <= N; k *= 2) {
    for (int j = k/2; j > 0; j /= 2) {
      int l = i ^ j;
      if (l > i)
	if ((((i & k) == 0) && (a[i] > a[l])) ||
	    (((i & k) != 0) && (a[i] < a[l]))) {	      
	  float c = a[i];
	  a[i] = a[l];
	  a[l] = c;
	}
      __syncthreads();
    }
  }
}

void host_sort(float * a, int N) {
    for (int k = 2; k <= N; k *= 2) {
      for (int j = k/2; j > 0; j /= 2) {

	for (int i = 0; i < N; i++) {

	  int l = i ^ j;
	  if (l > i)
	    if ((((i & k) == 0) && (a[i] > a[l])) ||
		(((i & k) != 0) && (a[i] < a[l]))) {	      
	      swap(a[i], a[l]);
	    }
	}
      }
    }
}

int main() {
  //INPUTS
  int N = 32;

  float *ha = new float[N];
  float *da;
  cudaMalloc((void **)&da, N*sizeof(float));

  // set problem input (b)
  for (int i = 0; i<N; ++i) {
    ha[i] = rand() % 1000;
  }
  
  cudaMemcpy(da, ha, N*sizeof(float), cudaMemcpyHostToDevice);

  sort<<<1,N>>>(da, N);

  cudaDeviceSynchronize();

  cudaMemcpy(ha, da, N*sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i<N; ++i) {
    printf("%f\n", ha[i]);
  }
  free(ha);
  return 0;
}
