#include <stdio.h>

__global__ void reduce_sum(int * da, int N) {
  int W = blockDim.x;
  int stride = W * 2;
  int tid = threadIdx.x;
  for(int i=tid+stride; i<N; i+=stride) da[tid]+=da[i];
  __syncthreads();


  for(int delta=1; delta<=W; delta*=2) {
    int i = tid*(2*delta);
    if (i + delta < N) {
      da[i] += da[i+delta];
      printf("%i (%i): %i\n", i, delta, da[i]);
    }
    __syncthreads();
  }
}

int main() {
  //INPUTS
  int N = 40;

  int *ha = new int[N];
  int *da;
  cudaMalloc((void **)&da, N*sizeof(int));

  // set problem input (b)
  for (int i = 0; i<N; ++i) {
    ha[i] = i*i;
  }
  
  cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);

  int W = 16;
  reduce_sum<<<1,W>>>(da, N);
  cudaDeviceSynchronize();

  int sum;
  cudaMemcpy(&sum, da, sizeof(int), cudaMemcpyDeviceToHost);

  int expected_sum = (N-1)*N*(2*N-1)/6;
  printf("%i (should be %i)", sum, expected_sum);
  cudaFree(da);
  free(ha);
  return 0;
}
