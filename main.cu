#include <stdio.h>

__global__ void f() {
  printf("Hello from %i, %i\n", blockIdx.x, threadIdx.x);
}

int main(void) {

f<<<5, 3>>>();
  cudaDeviceSynchronize();

  f<<<7, 2>>>();
  cudaDeviceSynchronize();

return 0;
}
