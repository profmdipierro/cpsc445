#include "math.h"
#include <cstdio>

float uniform() {
  return (rand() % 100000) / 100000;
}

float rand_in_range(float a, float b) {
    return (b - a) * uniform() + a;
}

float sigmoid(float x) {
  return tanh(x);
}

float dsigmoid(float y) {
  return 1.0 - y * y;
}

void update(float *ai, float *ah, float * ao,
	    float *wi, float *wo,
	    int ni, int nh, int no) {

  for(int j=0; j<nh; j++) {
    float sum=0.0;
    for(int i=0; i<ni; i++) sum += ai[i] * wi[i*nh+j];
    ah[j] = sigmoid(sum);
  }

  for(int k=0; k<no; k++) {
    float sum=0.0;
    for(int j=0; j<nh; j++) sum += ah[j] * wo[j*no+k];
    ao[k] = sigmoid(sum);
  }
}

float back_propagate(float * targets, float N, float M,
		     float *ai, float *ah, float * ao,
		     float *wi, float *wo, float *ci, float *co,
		     int ni, int nh, int no) {
  float *output_delta = new float[no];
  for(int i=0; i<no; i++) output_delta[i] = 0.0;

  for(int k=0; k<no; k++) {
    float error = targets[k] - ao[k];
    output_delta[k] = dsigmoid(ao[k]) * error;
  }

  float *hidden_delta = new float[nh];
  for(int i=0; i<no; i++) hidden_delta[i] = 0.0;

  for(int j=0; j<nh; j++) {
    float error = 0.0;
    for(int k=0; k<no; k++) {
      error += output_delta[k] * wo[j *no + k];
    }
    hidden_delta[j] = dsigmoid(ah[j]) * error;
  }
  
  for(int j=0; j<nh; j++) {
    for(int k=0; k<no; k++) {
      float change = output_delta[k] * ah[j];
      wo[j*no+ k] = wo[j*no+k] + N * change + M * co[j*no+k];
      co[j*no+ k] = change;
    }
  }

  for(int i=0; i<ni; i++) {
    for(int j=0; j<nh; j++) {
      float change = hidden_delta[j] * ai[i];
      wi[i*nh+ j] = wi[i*nh+ j] + N * change + M * ci[i*nh+ j];
      ci[i*nh+ j] = change;
    }
  }

  float error = 0.0;
  for(int k=0; k<no; k++) {
    error += pow(targets[k] - ao[k], 2);
  }

  delete[] output_delta;
  delete[] hidden_delta;
  return error;
}

struct data_point {
  float inputs[3];
  float expected[1];
};


void train(data_point *points, int np, int iterations, float N, float M,
	   float *ai, float *ah, float * ao,
	   float *wi, float *wo,
	   int ni, int nh, int no) {
  float *ci = new float[ni*nh];
  float *co = new float[nh*no];  

  for(int i=0; i<iterations; i++) {
    float error = 0.0;
    for(int k=0; k<np; k++) {
      for(int i=0; i<ni-1; i++) {
	ai[i] = points[k].inputs[i];
      }      
      update(ai, ah, ao, wi, wo, ni, nh, no);
      error += back_propagate(points[k].expected, N, M, ai, ah, ao, wi, wo, ci, co, ni, nh, no);
      printf("error=%f\n", error);
    }
  }

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


