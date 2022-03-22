#include <bits/stdc++.h>
#include "gen_data.h"

void generateTestDataDim(int p, int q, int numberOfCell,
    int &I, int &J, int &K, int &L, int &M, int geometryDim = 3, int topologyDim = 3) {
    I = q * (q+1) * (q+2) / 6;
    K = M = (p+1) * (p+2) * (p+3) / 6;
    J = numberOfCell;
    L = geometryDim;
    printf("Data benchmark:[I,J,K,L,M]=[%d,%d,%d,%d,%d]\n", I, J, K, L, M);
}

int main() {
    DATA_TYPE *a,*b,*c,*d,*e;
    int I,J,K,L,M;
    generateTestDataDim(3, 3, 100000, I, J, K, L, M);
    malloc_data(TENSOR_DATA);
    CHECK_TIME(opt_ijklm(TENSOR_DATA), "opt_ijklm");
    CHECK_TIME(opt_ijklm(TENSOR_DATA), "opt_ijklm");
    free_data(TENSOR_DATA);
    return 0;
}
