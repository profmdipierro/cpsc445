#include <stdio.h>

__global__ void f() {
  printf("(%i,%i) (%i,%i,%i)\n", blockIdx.x, blockIdx.y,
  threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(void) {

  dim2 A(7,7);
  dim3 B(4,4,3);
  
  f<<<A, B>>>();
  cudaDeviceSynchronize();

  return 0;
}
