#include <stdio.h>

__global__ void reduce_sum_step1(int * da, int N) {
  int B = gridDim.x;
  int W = blockDim.x;
  int shift = W * B;

  __shared__ int tmp[1024];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  tmp[tid] = da[gid];
  
  for(int i=gid+shift; i<N; i+=shift) {
    tmp[tid]+=da[i];
  }
  
  __syncthreads();
  
  for(int delta=1; delta<W; delta*=2) {    
    int i = threadIdx.x;
    if (i + delta < W) {
      tmp[i] += tmp[i+delta];
    }
    __syncthreads();
  }

  shift = blockDim.x * blockIdx.x;
  da[shift] = tmp[0];
}

__global__ void reduce_sum_step2(int * da, int W) {
  int B = blockDim.x;
  int shift = B;
  int tid = threadIdx.x;

  __shared__ int tmp[1024];  

  for(int i=0; i<W; i++)
    tmp[i] = da[i*W];
  
  for(int delta=1; delta<B; delta*=2) {    
    int i = tid*2*delta;
    if (i + delta < B) {
      tmp[i] += tmp[i+delta];
    }
    __syncthreads();
  }
  da[0] = tmp[0];
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
  cudaMemcpy(&sum, da, sizeof(int), cudaMemcpyDeviceToHost);

  int expected_sum =  (N-1)*N*(2*N-1)/6;
  printf("%i (should be %i)", sum, expected_sum);
  cudaFree(da);
  free(ha);
  free(hb);
  return 0;
}
