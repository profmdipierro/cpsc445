#include <stdio.h>

__global__ void f(double *da) {
  printf("(%i,%i) (%i,%i,%i)\n", blockIdx.x, blockIdx.y,
  threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(void) {

  cudaSetDevice(0);

  dim3 A(7,7,1);
  dim3 B(4,4,3);

  double *da;
  cudaMalloc(&da, 100*sizeof(float));

  f<<<A, B>>>(3);
  cudaDeviceSynchronize();

  return 0;
}
