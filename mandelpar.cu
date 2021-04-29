

#include <assert.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

#include "mapapixel.cuh"

#define TRUE  1
#define FALSE 0

static int dibujado = FALSE;
static tipoMapa miMapa;

#define FILAS    960
#define COLUMNAS 960

#define anchoBloque 16  // En 2D => 16x16 = 256 threads

// Coordenadas del mandelbrot inicial en miniatura
static double miAltura = 0.000305;
static double centroX  = -0.813997;
static double centroY  =  0.194129;

static int *colores, *coloresd; 
static int sizeColores;


static double planoVx; 
static double planoVy;

static short  planoFILAS;
static short  planoCOLUMNAS;
static double planoALTURA;
static double planoANCHURA;

static double planoFactorX;
static double planoFactorY;

//-------------------------------------------------------------------
void planoMapear (short filas, short columnas,
                  double Cx, double Cy, double laAltura)
{
  planoFILAS    = filas;
  planoCOLUMNAS = columnas;
  planoALTURA   = laAltura;
  planoANCHURA  = (planoCOLUMNAS * planoALTURA) 
  planoVx       = Cx - (planoANCHURA / 2.0);
  planoVy       = Cy - (planoALTURA  / 2.0);
  planoFactorX  = planoANCHURA / (double) (planoCOLUMNAS - 1);
  planoFactorY  = planoALTURA  / (double) (planoFILAS    - 1);
}

//-------------------------------------------------------------------
void planoPixelAPunto (short fila, short columna, double *X, double *Y) {
  *X = (planoFactorX * (double) columna) + planoVx;
  *Y = (planoFactorY * ((double) (planoFILAS - 1) - (double) fila)) + planoVy;
}

//--------------------------------------------------------------------
__global__ void mandelKernel(double planoFactorXd, double planoFactorYd,
                             double planoVxd,      double planoVyd, 
                             int maxIteracionesd,  int *coloresd)
{
  int fila, columna, i;
  double X, Y;
  double pReal = 0.0;    // Parte real       X
  double pImag = 0.0;    // Parte imaginaria Y
  double pRealAnt, pImagAnt, distancia;

  // Determinar pixel
  fila    = blockIdx.y*anchoBloque + threadIdx.y;
  columna = blockIdx.x*anchoBloque + threadIdx.x;

  // planoPixelAPunto
  X = (planoFactorXd * (double) columna) + planoVxd;
  Y = (planoFactorYd * ((double) (FILAS - 1) - (double) fila)) + planoVyd;

  // Se evalua la formula de Mandelbrot
  i=0;
  do {
    pRealAnt = pReal;
    pImagAnt = pImag;
    pReal = ((pRealAnt*pRealAnt) - (pImagAnt*pImagAnt)) + X;
    pImag = (2.0 * (pRealAnt*pImagAnt)) + Y;
    i++;
    distancia = pReal*pReal + pImag*pImag;
  } while ((i < maxIteracionesd) && (distancia <= 4.0));
  
  if (i == maxIteracionesd) i = 0;
  coloresd[fila * COLUMNAS + columna] = i;
}

//--------------------------------------------------------------------
static void dibujar()
{
  int fila, columna, i;
  tipoRGB colorRGB;
  struct timeval t0, t1, t;

  assert (gettimeofday(&t0, NULL) == 0);
  planoMapear (FILAS, COLUMNAS,
               centroX, centroY, miAltura);
  // Invocar al kernel
  dim3 dimGrid (FILAS/anchoBloque, COLUMNAS/anchoBloque);
  dim3 dimBlock(anchoBloque, anchoBloque);
  mandelKernel<<<dimGrid, dimBlock>>>(planoFactorX, planoFactorY,
                                      planoVx, planoVy,
                                      mapiNumColoresDefinidos(), coloresd);
  assert (cudaDeviceSynchronize() == 0);
  // Recoger matriz de colores
  assert (cudaMemcpy (colores, coloresd, sizeColores, cudaMemcpyDeviceToHost) == 0);
  i = 0;
  for (fila=0; fila<FILAS; fila++)
    for (columna=0; columna<COLUMNAS; columna++) {
      mapiColorRGB (colores[i++], &colorRGB);
      mapiPonerPuntoRGB (&miMapa, fila, columna, colorRGB);
    }
  mapiDibujarMapa(&miMapa);
  assert (gettimeofday(&t1, NULL) == 0);
  timersub(&t1, &t0, &t);
  printf("Tiempo => %ld:%ld (seg:mseg)\n", t.tv_sec, t.tv_usec/1000);
}

//--------------------------------------------------------------------
static void clickRaton (short fila, short columna, int botonIzquierdo)
{
  if (!dibujado) {
      dibujar();
      dibujado = TRUE;
  } else {
      planoPixelAPunto (fila, columna, &centroX, &centroY);
      if (botonIzquierdo)
        miAltura = miAltura / 2.0;  // Profundizar
      else
        miAltura = miAltura * 2.0;  // Alejarse
        dibujar ();
  }
}

//--------------------------------------------------------------------
static void finalizar ()
{
  cudaFree(coloresd);
}

//--------------------------------------------------------------------
int main(int argc, char *argv[]) {

  int profundidadColor;

  if      (argc == 1) profundidadColor = 3;
  else if (argc == 2) profundidadColor = atoi(argv[1]);
  else {
    printf ("Uso: mandelpar [profundidadColor]\n");
    return(0);
  }
  mapiProfundidadColor(profundidadColor);
  miMapa.elColor  = colorRGB;
  miMapa.filas    = FILAS;
  miMapa.columnas = COLUMNAS;
  mapiCrearMapa (&miMapa);
  
  sizeColores    = FILAS * COLUMNAS * sizeof(int);
  colores = (int *) malloc (sizeColores);
  // Ubico matriz de colores en GPU
  assert (cudaMalloc ((void**) &coloresd,  sizeColores   ) == 0);
  mapiInicializar (FILAS, COLUMNAS, dibujar, clickRaton, finalizar);
  return 0;
}

