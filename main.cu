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

__global__ void reduce_sum_step2(int * da, int W) {
  int B = blockDim.x;
  int shift = B;
  int tid = threadIdx.x;
  for(int delta=1; delta<B; delta*=2) {    
    int i = tid*2*delta;
    if (i + delta < B) {
      da[i*W] += da[(i+delta)*W];
    }
    __syncthreads();
  }  
}

int main() {
  //INPUTS
  int N = 1000;

  int *ha = new int[N];
  int *hb = new int[N];
  int *da;
  cudaMalloc((void **)&da, N*sizeof(int));

  // set problem input (b)
  for (int i = 0; i<N; ++i) {
    ha[i] = i*i;
  }
  
  cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);

  int B = 3;
  int W = 16;
  reduce_sum_step1<<<B,W>>>(da, N);
  cudaDeviceSynchronize();
  reduce_sum_step2<<<1,B>>>(da, W);
  cudaDeviceSynchronize();

  int sum;
  // cudaMemcpy(&sum, da, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(hb, da, N*sizeof(int), cudaMemcpyDeviceToHost);

  printf("%i\n", hb[0]);
  // printf("%i\n", hb[16]);
  // printf("%i\n", hb[32]);
  // printf("%i\n", hb[32]);

  sum = hb[0]; //  + hb[32];
  int expected_sum =  (N-1)*N*(2*N-1)/6;
  printf("%i (should be %i)", sum, expected_sum);
  cudaFree(da);
  free(ha);
  free(hb);
  return 0;
}
