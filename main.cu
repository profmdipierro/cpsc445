#include <stdio.h>

__global__ void reduce_sum_step1(int * da, int N) {
  int B = gridDim.x;
  int W = blockDim.x;
  int shift = W * B;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  for(int i=gid+shift; i<N; i+=shift) {
    da[gid]+=da[i];
  }
  
  __syncthreads();
  
  shift = blockDim.x * blockIdx.x;
  for(int delta=1; delta<W; delta*=2) {    
    int i = threadIdx.x;
    if (i + delta < W) {
      da[i+shift] += da[i+shift+delta];
    }
    __syncthreads();
  }  
}

/*
__global__ void reduce_sum(int * da, int N) {
  int B = gridDim.x;
  int W = blockDim.x;
  int shift = W * B;

  int gid = blockIdx.x* blockDim.x + threadIdx.x;
  
  for(int i=gid+shift; i<N; i+=shift) da[gid]+=da[i];
}
*/

int main() {
  //INPUTS
  int N = 100;

  int *ha = new int[N];
  int *hb = new int[N];
  int *da;
  cudaMalloc((void **)&da, N*sizeof(int));

  // set problem input (b)
  for (int i = 0; i<N; ++i) {
    ha[i] = i;
  }
  
  cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);

  int B = 3;
  int W = 8;
  reduce_sum_step1<<<B,W>>>(da, N);
  cudaDeviceSynchronize();

  int sum;
  // cudaMemcpy(&sum, da, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(hb, da, N*sizeof(int), cudaMemcpyDeviceToHost);

  printf("%i\n", hb[0]);
  printf("%i\n", hb[8]);
  printf("%i\n", hb[16]);
  // printf("%i\n", hb[32]);

  sum = hb[0] + hb[8] + hb[16]; //  + hb[32];
  int expected_sum = (N-1)*N/2; // (N-1)*N*(2*N-1)/6;
  printf("%i (should be %i)", sum, expected_sum);
  cudaFree(da);
  free(ha);
  free(hb);
  return 0;
}
