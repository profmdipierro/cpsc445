#include <stdio.h>

__global__ void reduce_sum(float * da, int N) {
  int W = blockDim.x;
  int tid = threadIdx.x;
  for(int i=tid+W; i<N; i+=W)  da[tid] += da[i];
}

int main() {
  //INPUTS
  int N = 1000;
    
  int *ha = new int[N];
  int *da;
  cudaMalloc((void **)&da, N*sizeof(int));

  // set problem input (b)
  for (int i = 0; i<N; ++i) {
    ha[i] = i*i;
  }
  
  cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);

  int W = 10;
  reduce_sum<<<1,W>>>(da, N);
  cudaDeviceSynchronize();

  int sums[10];
  cudaMemcpy(sums, da, W*sizeof(int), cudaMemcpyDeviceToHost);

  int sum=0;
  for(int i=0; i<W; i++) sum+=sums[i];
  
  printf("%i", sum);
  cudaFree(da);
  free(ha);
  return 0;
}
