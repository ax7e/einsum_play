#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <cutensor.h>

#include <unordered_map>
#include <vector>

// Handle cuTENSOR errors
#define HANDLE_ERROR(x) {                                                              \
  const auto err = x;                                                                  \
  if( err != CUTENSOR_STATUS_SUCCESS )                                                 \
  { printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__); exit(-1); } \
}
using std::vector; 

void generateTestDataDim(int p, int q, int numberOfCell,
    int64_t &I, int64_t &J, int64_t &K, int64_t &L, int64_t &M, int geometryDim = 3, int topologyDim = 3) {
    I = q * (q+1) * (q+2) / 6;
    K = M = (p+1) * (p+2) * (p+3) / 6;
    J = numberOfCell;
    L = geometryDim;
    printf("Data benchmark:[I,J,K,L,M]=[%d,%d,%d,%d,%d]\n", I, J, K, L, M);
}

typedef float floatType;
typedef CUDA_R_32F tensorType;
typedef float floatTypeCompute;

void initTenosr(const vector<int> &mode, int &nmode, const vector<int> &extentLib, std::unordered_map<int, int64_t> &extent, size_t &eleCnt, size_t &sizeByte, void *&T_d, floatType *& T_h, 
    cutensorHandle_t &handle, cutensorTensorDescriptor_t &desc, uint32_t &align) {
    nmode = mode.size(); 
    for(auto m : mode) extent.push_back(extentLib[m]);
    eleCnt = 1; 
    for(auto m : mode) eleCnt *= extentLib[m];
    sizeByte = sizeof(floatType) * eleCnt;
    cudaMalloc((void**)&T_d, sizeByte);
    T_h = (floatType*) malloc(sizeof(floatType) * eleCnt);
    for (int64_t i = 0; i < elementsA; i++)
      T_h[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    cudaMemcpy(T_d, T_h, sizeByte, cudaMemcpyHostToDevice);
    HANDLE_ERROR( cutensorInitTensorDescriptor( &handle,
      &desc,
      nmode,
      extent.data(),
      NULL,/*stride*/
      tensorType, CUTENSOR_OP_IDENTITY ) ); 
    HANDLE_ERROR( cutensorGetAlignmentRequirement( &handle,
       T_d,
       &desc,
       &align) ); 
}

int main(int argc, char** argv)
{
  //e(j,k,m)=a(i)*b(i,j,k,l)*c(i,j,m,l)*d(j)
  //e(j,k,m)=a(i)*f(i,j,k,m,l)*d(j)
  //e(j,k,m)=g(i,j)*f(i,j,k,m,l)

  // CUDA types
  cudaDataType_t type = CUDA_R_32F;
  cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;

  printf("Include headers and define data types\n");

  /* ***************************** */

  // Create vector of modes
  std::vector<int> modeA{'i'}, 
                   modeB{'i','j','k','l'}, 
                   modeC{'i','j','m','l'}, 
                   modeD{'j'}, 
                   modeE{'j','k','m'}, 
                   modeF{'i','j','k','m','l'}, 
                   modeG{'i','j'};
  int nmodeA, nmodeB, nmodeC, nmodeD, nmodeE, nmodeF, nmodeG;

  // Extents
  std::unordered_map<int, int64_t> extent;
  generateTestDataDim(3, 3, 100, extent['i'], extent['j'], extent['k'], extent['m'], extent['l']);

  // Create a vector of extents for each tensor
  std::vector<int64_t> extentG, extentF, extentE, extentD, extentC, extentA, extentB;
  // Number of elements of each tensor
  size_t elementsA,elementsB,elementsC,elementsD,elementsE,elementsF,elementsG;


  // Size in bytes
  size_t sizeA,sizeB,sizeC,sizeD,sizeE,sizeF,sizeG;

  // Allocate on device
  void *A_d, *B_d, *C_d, *D_d, *E_d, *F_d, *G_d;

  // Allocate on host
  floatTypeA *A_h,*B_h,*C_h,*D_h,*E_h,*F_h,*G_h;

  // Initialize cuTENSOR library
  cutensorHandle_t handle;
  cutensorInit(&handle);

  // Create Tensor Descriptors
  cutensorTensorDescriptor_t descA,descB,descC,descD,descE,descF,descG;

  uint32_t alignA,alignB,alignC,alignD,alignE,alignF,alignG;
  initTenosr(modeA,nmodeA,extent,extentA,elementsA,sizeA,A_d,A_h,handle,descA,alignA);
  initTenosr(modeB,nmodeB,extent,extentB,elementsB,sizeB,B_d,B_h,handle,descB,alignB);
  initTenosr(modeC,nmodeC,extent,extentC,elementsC,sizeC,C_d,C_h,handle,descC,alignC);
  initTenosr(modeD,nmodeD,extent,extentD,elementsD,sizeD,D_d,D_h,handle,descD,alignD);
  initTenosr(modeE,nmodeE,extent,extentE,elementsE,sizeE,E_d,E_h,handle,descE,alignE);
  initTenosr(modeF,nmodeF,extent,extentF,elementsF,sizeF,F_d,F_h,handle,descF,alignF);
  initTenosr(modeG,nmodeG,extent,extentG,elementsG,sizeG,G_d,G_h,handle,descG,alignG);
  // Create the Contraction Descriptor
  cutensorContractionDescriptor_t desc;
  HANDLE_ERROR( cutensorInitContractionDescriptor( &handle,
              &desc,
              &descA, modeA.data(), alignmentRequirementA,
              &descB, modeB.data(), alignmentRequirementB,
              &descC, modeC.data(), alignmentRequirementC,
              &descC, modeC.data(), alignmentRequirementC,
              typeCompute) );

  printf("Initialize contraction descriptor\n");

  /* ***************************** */

  // Set the algorithm to use
  cutensorContractionFind_t find;
  HANDLE_ERROR( cutensorInitContractionFind(
              &handle, &find,
              CUTENSOR_ALGO_DEFAULT) );

  printf("Initialize settings to find algorithm\n");

  /* ***************************** */

  // Query workspace
  size_t worksize = 0;
  HANDLE_ERROR( cutensorContractionGetWorkspace(&handle,
              &desc,
              &find,
              CUTENSOR_WORKSPACE_RECOMMENDED, &worksize ) );

  // Allocate workspace
  void *work = nullptr;
  if(worksize > 0)
  {
      if( cudaSuccess != cudaMalloc(&work, worksize) ) // This is optional!
      {
          work = nullptr;
          worksize = 0;
      }
  }

  printf("Query recommended workspace size and allocate it\n");

  /* ***************************** */

  // Create Contraction Plan
  cutensorContractionPlan_t plan;
  HANDLE_ERROR( cutensorInitContractionPlan(&handle,
                                            &plan,
                                            &desc,
                                            &find,
                                            worksize) );

  printf("Create plan for contraction\n");

  /* ***************************** */

  cutensorStatus_t err;

  // Execute the tensor contraction
  err = cutensorContraction(&handle,
                            &plan,
                     (void*)&alpha, A_d,
                                    B_d,
                     (void*)&beta,  C_d,
                                    C_d,
                            work, worksize, 0 /* stream */);
  cudaDeviceSynchronize();

  // Check for errors
  if(err != CUTENSOR_STATUS_SUCCESS)
  {
      printf("ERROR: %s\n", cutensorGetErrorString(err));
  }

  printf("Execute contraction from plan\n");

  /* ***************************** */

  if ( A ) free( A );
  if ( B ) free( B );
  if ( C ) free( C );
  if ( A_d ) cudaFree( A_d );
  if ( B_d ) cudaFree( B_d );
  if ( C_d ) cudaFree( C_d );
  if ( work ) cudaFree( work );

  printf("Successful completion\n");

  return 0;
}