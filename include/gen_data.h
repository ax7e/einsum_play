#include <bits/stdc++.h>
#define lop(i,s,t) for(int i=s;i<(t);++i)
#define DATA_TYPE double
#define TENSOR_ARG int I, int J, int K, int L, int M, DATA_TYPE *&a, DATA_TYPE *&b, DATA_TYPE *&c, DATA_TYPE *&d, DATA_TYPE *&e
#define TENSOR_DATA I,J,K,L,M,a,b,c,d,e

using namespace std;
using namespace chrono;
#define CHECK_TIME(f, msg) \
{\
	auto start = high_resolution_clock::now();\
	f;\
	auto stop = high_resolution_clock::now();\
	auto duration = duration_cast<milliseconds>(stop - start);\
	cout << msg << ":" << duration.count() << "ms" << endl;\
}\


//a[i]*b[i][j][k][l]*c[i][j][m][l]*d[j]=e[j][k][m]
void malloc_data(TENSOR_ARG) {
    a = (DATA_TYPE*) malloc(sizeof(DATA_TYPE) * I);
    b = (DATA_TYPE*) malloc(sizeof(DATA_TYPE) * I * J * K * L);
    c = (DATA_TYPE*) malloc(sizeof(DATA_TYPE) * I * J * M * L);
    d = (DATA_TYPE*) malloc(sizeof(DATA_TYPE) * J);
    e = (DATA_TYPE*) malloc(sizeof(DATA_TYPE) * J * K * M);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> val_gen(-10.0, 10.0);
    lop(i,0,I)a[i]=val_gen(gen);
    lop(i,0,I*J*K*L)b[i]=val_gen(gen);
    lop(i,0,I*J*M*L)c[i]=val_gen(gen);
    lop(i,0,J)d[i]=val_gen(gen);
    lop(i,0,J*K*M)e[i]=val_gen(gen);
}

void free_data(TENSOR_ARG) {
    free(a), free(b), free(c), free(d), free(e);
}

void opt_ijklm(TENSOR_ARG){
    lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;
    lop(i,0,I)
        #pragma omp parallel for
        lop(j,0,J)
            lop(k,0,K)
                lop(l,0,L)
                    lop(m,0,M)
                        e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];
}

void ijklm(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(j,0,J)lop(k,0,K)lop(l,0,L)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ijkml(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(j,0,J)lop(k,0,K)lop(m,0,M)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ijlkm(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(j,0,J)lop(l,0,L)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ijlmk(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(j,0,J)lop(l,0,L)lop(m,0,M)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ijmkl(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(j,0,J)lop(m,0,M)lop(k,0,K)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ijmlk(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(j,0,J)lop(m,0,M)lop(l,0,L)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ikjlm(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(k,0,K)lop(j,0,J)lop(l,0,L)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ikjml(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(k,0,K)lop(j,0,J)lop(m,0,M)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ikljm(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(k,0,K)lop(l,0,L)lop(j,0,J)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void iklmj(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(k,0,K)lop(l,0,L)lop(m,0,M)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ikmjl(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(k,0,K)lop(m,0,M)lop(j,0,J)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ikmlj(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(k,0,K)lop(m,0,M)lop(l,0,L)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void iljkm(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(l,0,L)lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void iljmk(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(l,0,L)lop(j,0,J)lop(m,0,M)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ilkjm(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(l,0,L)lop(k,0,K)lop(j,0,J)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ilkmj(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(l,0,L)lop(k,0,K)lop(m,0,M)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ilmjk(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(l,0,L)lop(m,0,M)lop(j,0,J)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ilmkj(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(l,0,L)lop(m,0,M)lop(k,0,K)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void imjkl(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(m,0,M)lop(j,0,J)lop(k,0,K)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void imjlk(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(m,0,M)lop(j,0,J)lop(l,0,L)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void imkjl(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(m,0,M)lop(k,0,K)lop(j,0,J)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void imklj(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(m,0,M)lop(k,0,K)lop(l,0,L)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void imljk(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(m,0,M)lop(l,0,L)lop(j,0,J)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void imlkj(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(i,0,I)lop(m,0,M)lop(l,0,L)lop(k,0,K)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jiklm(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(i,0,I)lop(k,0,K)lop(l,0,L)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jikml(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(i,0,I)lop(k,0,K)lop(m,0,M)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jilkm(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(i,0,I)lop(l,0,L)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jilmk(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(i,0,I)lop(l,0,L)lop(m,0,M)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jimkl(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(i,0,I)lop(m,0,M)lop(k,0,K)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jimlk(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(i,0,I)lop(m,0,M)lop(l,0,L)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jkilm(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(k,0,K)lop(i,0,I)lop(l,0,L)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jkiml(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(k,0,K)lop(i,0,I)lop(m,0,M)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jklim(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(k,0,K)lop(l,0,L)lop(i,0,I)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jklmi(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(k,0,K)lop(l,0,L)lop(m,0,M)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jkmil(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(k,0,K)lop(m,0,M)lop(i,0,I)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jkmli(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(k,0,K)lop(m,0,M)lop(l,0,L)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jlikm(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(l,0,L)lop(i,0,I)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jlimk(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(l,0,L)lop(i,0,I)lop(m,0,M)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jlkim(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(l,0,L)lop(k,0,K)lop(i,0,I)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jlkmi(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(l,0,L)lop(k,0,K)lop(m,0,M)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jlmik(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(l,0,L)lop(m,0,M)lop(i,0,I)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jlmki(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(l,0,L)lop(m,0,M)lop(k,0,K)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jmikl(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(m,0,M)lop(i,0,I)lop(k,0,K)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jmilk(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(m,0,M)lop(i,0,I)lop(l,0,L)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jmkil(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(m,0,M)lop(k,0,K)lop(i,0,I)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jmkli(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(m,0,M)lop(k,0,K)lop(l,0,L)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jmlik(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(m,0,M)lop(l,0,L)lop(i,0,I)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void jmlki(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(j,0,J)lop(m,0,M)lop(l,0,L)lop(k,0,K)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kijlm(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(i,0,I)lop(j,0,J)lop(l,0,L)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kijml(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(i,0,I)lop(j,0,J)lop(m,0,M)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kiljm(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(i,0,I)lop(l,0,L)lop(j,0,J)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kilmj(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(i,0,I)lop(l,0,L)lop(m,0,M)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kimjl(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(i,0,I)lop(m,0,M)lop(j,0,J)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kimlj(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(i,0,I)lop(m,0,M)lop(l,0,L)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kjilm(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(j,0,J)lop(i,0,I)lop(l,0,L)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kjiml(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(j,0,J)lop(i,0,I)lop(m,0,M)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kjlim(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(j,0,J)lop(l,0,L)lop(i,0,I)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kjlmi(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(j,0,J)lop(l,0,L)lop(m,0,M)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kjmil(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(j,0,J)lop(m,0,M)lop(i,0,I)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kjmli(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(j,0,J)lop(m,0,M)lop(l,0,L)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void klijm(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(l,0,L)lop(i,0,I)lop(j,0,J)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void klimj(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(l,0,L)lop(i,0,I)lop(m,0,M)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kljim(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(l,0,L)lop(j,0,J)lop(i,0,I)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kljmi(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(l,0,L)lop(j,0,J)lop(m,0,M)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void klmij(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(l,0,L)lop(m,0,M)lop(i,0,I)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void klmji(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(l,0,L)lop(m,0,M)lop(j,0,J)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kmijl(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(m,0,M)lop(i,0,I)lop(j,0,J)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kmilj(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(m,0,M)lop(i,0,I)lop(l,0,L)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kmjil(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(m,0,M)lop(j,0,J)lop(i,0,I)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kmjli(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(m,0,M)lop(j,0,J)lop(l,0,L)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kmlij(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(m,0,M)lop(l,0,L)lop(i,0,I)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void kmlji(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(k,0,K)lop(m,0,M)lop(l,0,L)lop(j,0,J)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void lijkm(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(i,0,I)lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void lijmk(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(i,0,I)lop(j,0,J)lop(m,0,M)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void likjm(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(i,0,I)lop(k,0,K)lop(j,0,J)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void likmj(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(i,0,I)lop(k,0,K)lop(m,0,M)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void limjk(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(i,0,I)lop(m,0,M)lop(j,0,J)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void limkj(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(i,0,I)lop(m,0,M)lop(k,0,K)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ljikm(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(j,0,J)lop(i,0,I)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ljimk(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(j,0,J)lop(i,0,I)lop(m,0,M)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ljkim(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(j,0,J)lop(k,0,K)lop(i,0,I)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ljkmi(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(j,0,J)lop(k,0,K)lop(m,0,M)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ljmik(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(j,0,J)lop(m,0,M)lop(i,0,I)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void ljmki(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(j,0,J)lop(m,0,M)lop(k,0,K)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void lkijm(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(k,0,K)lop(i,0,I)lop(j,0,J)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void lkimj(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(k,0,K)lop(i,0,I)lop(m,0,M)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void lkjim(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(k,0,K)lop(j,0,J)lop(i,0,I)lop(m,0,M)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void lkjmi(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(k,0,K)lop(j,0,J)lop(m,0,M)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void lkmij(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(k,0,K)lop(m,0,M)lop(i,0,I)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void lkmji(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(k,0,K)lop(m,0,M)lop(j,0,J)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void lmijk(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(m,0,M)lop(i,0,I)lop(j,0,J)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void lmikj(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(m,0,M)lop(i,0,I)lop(k,0,K)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void lmjik(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(m,0,M)lop(j,0,J)lop(i,0,I)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void lmjki(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(m,0,M)lop(j,0,J)lop(k,0,K)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void lmkij(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(m,0,M)lop(k,0,K)lop(i,0,I)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void lmkji(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(l,0,L)lop(m,0,M)lop(k,0,K)lop(j,0,J)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mijkl(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(i,0,I)lop(j,0,J)lop(k,0,K)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mijlk(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(i,0,I)lop(j,0,J)lop(l,0,L)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mikjl(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(i,0,I)lop(k,0,K)lop(j,0,J)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void miklj(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(i,0,I)lop(k,0,K)lop(l,0,L)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void miljk(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(i,0,I)lop(l,0,L)lop(j,0,J)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void milkj(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(i,0,I)lop(l,0,L)lop(k,0,K)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mjikl(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(j,0,J)lop(i,0,I)lop(k,0,K)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mjilk(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(j,0,J)lop(i,0,I)lop(l,0,L)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mjkil(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(j,0,J)lop(k,0,K)lop(i,0,I)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mjkli(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(j,0,J)lop(k,0,K)lop(l,0,L)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mjlik(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(j,0,J)lop(l,0,L)lop(i,0,I)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mjlki(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(j,0,J)lop(l,0,L)lop(k,0,K)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mkijl(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(k,0,K)lop(i,0,I)lop(j,0,J)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mkilj(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(k,0,K)lop(i,0,I)lop(l,0,L)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mkjil(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(k,0,K)lop(j,0,J)lop(i,0,I)lop(l,0,L)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mkjli(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(k,0,K)lop(j,0,J)lop(l,0,L)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mklij(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(k,0,K)lop(l,0,L)lop(i,0,I)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mklji(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(k,0,K)lop(l,0,L)lop(j,0,J)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mlijk(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(l,0,L)lop(i,0,I)lop(j,0,J)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mlikj(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(l,0,L)lop(i,0,I)lop(k,0,K)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mljik(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(l,0,L)lop(j,0,J)lop(i,0,I)lop(k,0,K)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mljki(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(l,0,L)lop(j,0,J)lop(k,0,K)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mlkij(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(l,0,L)lop(k,0,K)lop(i,0,I)lop(j,0,J)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}
void mlkji(TENSOR_ARG){lop(j,0,J)lop(k,0,K)lop(m,0,M)e[j*K*M+k*M+m]=0;lop(m,0,M)lop(l,0,L)lop(k,0,K)lop(j,0,J)lop(i,0,I)e[j*K*M+k*M+m]+=a[i]*b[i*J*K*L+j*K*L+k*L+l]*c[i*J*M*L+j*M*L+m*L+l]*d[j];}