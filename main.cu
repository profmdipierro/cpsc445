#include "math.h"
#include <cstdio>

float uniform() {
  return (rand() % 100000) / 100000;
}

float rand_in_range(float a, float b) {
    return (b - a) * uniform() + a;
}

__device__ float sigmoid(float x) {
  return tanh(x);
}

__device__ float dsigmoid(float y) {
  return 1.0 - y * y;
}


// all inputs are on device
__global__ void update(float *ai, float *ah, float * ao,
		       float *wi, float *wo,
	    int ni, int nh, int no) {
  int window = blockDim.x;
  int tid = threadIdx.x;
  for(int j=tid; j<nh; j+=window) {
    float sum=0.0;
    for(int i=0; i<ni; i++) sum += ai[i] * wi[i*nh+j];
    ah[j] = sigmoid(sum);
  }

  __syncthreads();
  
  for(int k=tid; k<no; k+=window) {
    float sum=0.0;
    for(int j=0; j<nh; j++) sum += ah[j] * wo[j*no+k];
    ao[k] = sigmoid(sum);
  }
}

__global__
void back_propagate(float N, float M,
		    float *ai, float *ah, float * ao,
		    float *wi, float *wo, float *ci, float *co,
		    float *output_delta, float *hidden_delta,
		    int ni, int nh, int no) {

  int window = blockDim.x;
  int tid = threadIdx.x;
  
  for(int k=tid; k<no; k+=window) {
    float error = targets[k] - ao[k];
    output_delta[k] = dsigmoid(ao[k]) * error;
  }

  __syncthreads();
  
  for(int j=tid; j<nh; j+=window) {
    float error = 0.0;
    for(int k=0; k<no; k++) {
      error += output_delta[k] * wo[j *no + k];
    }
    hidden_delta[j] = dsigmoid(ah[j]) * error;
  }
  
  __syncthreads();
  
  for(int j=tid; j<nh; j+=window) {
    for(int k=0; k<no; k++) {
      float change = output_delta[k] * ah[j];
      wo[j*no+ k] = wo[j*no+k] + N * change + M * co[j*no+k];
      co[j*no+ k] = change;
    }
  }

  __syncthreads();
  
  for(int i=tid; i<ni; i+=window) {
    for(int j=0; j<nh; j++) {
      float change = hidden_delta[j] * ai[i];
      wi[i*nh+ j] = wi[i*nh+ j] + N * change + M * ci[i*nh+ j];
      ci[i*nh+ j] = change;
    }
  }

}

struct data_point {
  float inputs[3];
  float expected[1];
};


// all inputs are on host
void train(data_point *points, int np, int iterations, float N, float M,
	   float *ai, float *ah, float * ao,
	   float *wi, float *wo,
	   int ni, int nh, int no, int nthreads) {

  float *ci=new float[ni*nh];
  float *co=new float[nh*no];
  float *dai;
  float *dah;
  float *dao;
  float *dwi;
  float *dwo;
  float *dci;
  float *dco;
  float *output_delta;
  float *hidden_delta;
  
  cudaMalloc((void**) &dai,ni*sizeof(float));
  cudaMalloc((void**) &dah,nh*sizeof(float));
  cudaMalloc((void**) &dao,no*sizeof(float));
  cudaMalloc((void**) &dwi,ni*nh*sizeof(float));
  cudaMalloc((void**) &dwo,nh*no*sizeof(float));
  
  cudaMalloc((void**) &dci,ni*nh*sizeof(float));
  cudaMalloc((void**) &dco,nh*no*sizeof(float));
  cudaMalloc((void**) &hidden_delta, nh*sizeof(float));
  cudaMalloc((void**) &output_delta, no*sizeof(float));
  // initiaize ci and co and copy inputs from host
  for(int k=0; k<ni*nh; k++) ci[k] = 0;
  for(int k=0; k<nh*no; k++) co[k] = 0;
  cudaMemcpy(dai, ai, ni*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dah, ah, nh*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dao, ao, no*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dwi, wi, ni*nh*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dwo, wo, nh*no*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dci, ci, ni*nh*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dco, co, nh*no*sizeof(float), cudaMemcpyHostToDevice);
  
  for(int i=0; i<iterations; i++) {
    float error;
    for(int k=0; k<np; k++) {
      /// copy
      for(int i=0; i<ni-1; i++) {
	ai[i] = points[k].inputs[i];
      }
      ///
      update<<<1, nthreads>>>(dai, dah, dao, dwi, dwo, ni, nh, no);
      back_propagate<<<1, nthreads>>>(N, M, dai, dah, dao, dwi, dwo, dci, dco, output_delta, hidden_delta, ni, nh, no);

      cudaMemcpy(ao, dao, no*sizeof(float), cudaMemcpyDeviceToHost);      
      cudaMemcpy(wi, dwi, ni*nh*sizeof(float), cudaMemcpyDeviceToHost);      
      cudaMemcpy(wo, dwo, nh*no*sizeof(float), cudaMemcpyDeviceToHost);      
      printf("error=%f\n", error);
      // copy dao -> ao
      error = 0.0;
      for(int k=0; k<no; k++) {
	error += pow(points[k].expected[k] - ao[k], 2);
      }
    }
  }

  cudaFree(dai);
  cudaFree(dah);
  cudaFree(dao);
  cudaFree(dwi);
  cudaFree(dwo);

  cudaFree(dci);
  cudaFree(dco);
  cudaFree(output_delta);
  cudaFree(hidden_delta);

  delete[] ci;
  delete[] co;
}

int main() {

  int num_inputs = 3;
  int num_outputs = 1;
  int num_hidden_layers = 4;
  
  int ni = num_inputs + 1;
  int nh = num_hidden_layers;
  int no = num_outputs;

  float *ai = new float[ni];
  float *ah = new float[nh];
  float *ao = new float[no];

  for(int i =0; i<ni; i++) ai[i]=1;
  for(int i =0; i<nh; i++) ah[i]=1;
  for(int i =0; i<no; i++) ao[i]=1;

  float *wi = new float[ni*nh];
  float *wo = new float[nh*no];
  for(int i =0; i<ni*nh; i++) wi[i]=rand_in_range(-0.2, +0.2);
  for(int i =0; i<nh*no; i++) wi[i]=rand_in_range(-2.0, +2.0);

  
  
  delete[] ai;
  delete[] ah;
  delete[] ao;
  delete[] wi;
  delete[] wo;
  return 0;
};


