#include <stdio.h>

__device__ void g() {
  printf("Hello from %i, %i\n", blockIdx.x, threadIdx.x);
}

__global__ void f() {
  g();  
}

int main(void) {

  f<<<5, 3>>>();
  cudaDeviceSynchronize();

return 0;
}
