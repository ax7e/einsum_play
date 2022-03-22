#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <cutensor.h>

int main(int argc, char** argv)
{
  // Host element type definition
  typedef float floatTypeA;
  typedef float floatTypeB;
  typedef float floatTypeC;
  typedef float floatTypeCompute;

  // CUDA types
  cudaDataType_t typeA = CUDA_R_32F;
  cudaDataType_t typeB = CUDA_R_32F;
  cudaDataType_t typeC = CUDA_R_32F;
  cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;

  floatTypeCompute alpha = (floatTypeCompute)1.1f;
  floatTypeCompute beta  = (floatTypeCompute)0.9f;

  printf("Include headers and define data types\n");

  return 0;
}
