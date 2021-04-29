

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define anchoBloque 16  
//-------------------------------------------------------------------
void initMatriz (float *M, int card, float valor) {
  int i;
  for (i=0; i<card; i++) M[i] = valor;
}

//-------------------------------------------------------------------
__global__ void mulMatrizKernel (float *Ad, float *Bd, float *Cd, int card) {
  int ROW= blockIdx.x*blockDim.x+threadIdx.x;
  int COL= blockIdx.y*blockDim.y+threadIdx.y;
  float tmpSum=0;
 
  if(ROW < card && COL < card){ 
for (int i = 0; i<card; i++){
   tmpSum = tmpSum + Ad[ROW*card+i] * Bd[i*card + COL]; }
}
 Cd[ROW*card+COL] = tmpSum;
}
//-------------------------------------------------------------------
int main (int argc, char *argv[])
{
  int filA, colA, filB, colB, filC, colC;
  struct timeval t0, tf, t;
  float  *A, *B, *C;
  float  *Ad, *Bd, *Cd;
  int    sizeA, sizeB, sizeC, k;

  filA = atoi(argv[1]);
  colA = filA;
  filB = filA;
  colB = filA;
  filC = filA;
  colC = filA;
  sizeA = filA*colA*sizeof(float);
  sizeB = filB*colB*sizeof(float);
  sizeC = filC*colC*sizeof(float);
  A = (float *) malloc (sizeA);
  B = (float *) malloc (sizeB);
  C = (float *) malloc (sizeC);
  initMatriz (A, filA*colA, 1.0f );
  initMatriz (B, filB*colB, 0.01f);

  assert (gettimeofday (&t0, NULL) == 0);
  
  cudaMalloc ((void**) &Ad, sizeA);
  cudaMemcpy (Ad, A, sizeA, cudaMemcpyHostToDevice);
  cudaMalloc ((void**) &Bd, sizeB);
  cudaMemcpy (Bd, B, sizeB, cudaMemcpyHostToDevice);
  
  cudaMalloc ((void**) &Cd, sizeC);
  
  dim3 dimGrid (filA/anchoBloque, filA/anchoBloque);
  dim3 dimBlock(anchoBloque, anchoBloque);
  mulMatrizKernel<<<dimGrid, dimBlock>>>(Ad, Bd, Cd, filA);
  cudaDeviceSynchronize();
 
  cudaMemcpy (C, Cd, sizeC, cudaMemcpyDeviceToHost);
  
  cudaFree(Ad); cudaFree(Bd); cudaFree(Cd);
  assert (gettimeofday (&tf, NULL) == 0);

  timersub (&tf, &t0, &t);
  printf ("Tiempo = %ld:%ld \n", t.tv_sec, t.tv_usec);
  printf ("C[0] = %f\n", C[filC-1]);
  for (k=1; k<(filC*colC); k++) assert (C[k] == C[k-1]);
  return 0;
}
