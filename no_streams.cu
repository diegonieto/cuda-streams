/*
 * Created by Diego Nieto Muñoz
 *
 * Approach:
 * We want to compute two kernels which have two inputs and one
 * output each. I will show the benefits of using asynchronous
 * copies.
 * The architecture Fermi has three streams:
 *   -> Host To Device copies
 *   -> Kernel computation
 *   -> Device To Host copies
 *
 * Definitions:
 *   -> operation1(d1_a, d1_b, d1_c)
 *   -> operation2(d2_a, d2_b, d2_c)
 *
 *   -> 6 streams
 *   -> 4 events
 *
 * Dependencies:
 *   stream1 -> d1_a, d1_b
 *   stream2 -> operation1 | event1 -> wait for stream1
 *   stream3 -> d1_c | event2 -> wait for stream2
 *
 *   stream4 -> d2_a, d2_b
 *   stream5 -> operation2 | event3 -> wait for stream4
 *   stream6 -> d2_c | event4 -> wait for stream5 
 *
 * Flow (2 iterations):
 *   H2D          1_a, 1_b  2_a, 2_b 
 *   Computation            operation1 operation2
 *   D2H                               1_c        2_c
 */

typedef float TYPE;

// Kernel 1
__global__
void operation1(TYPE *a,
                TYPE *b,
                TYPE *c,
                int n)
{
  unsigned int i, gid = blockIdx.x*blockDim.x+threadIdx.x;
  for(i = 0; i<25; i++)
    c[gid] = sqrt(a[gid]) * sqrt(b[gid]);
}

// Kernel 2
__global__
void operation2(TYPE *a,
                TYPE *b,
                TYPE *c,
                int n)
{
  unsigned int i, gid = blockIdx.x*blockDim.x+threadIdx.x;
  for(i = 0; i<5; i++)
    c[gid] = cos(a[gid]) * sin(b[gid]);
}

// Main program
int main(int argc, char *argv[]) {

  unsigned int i, n = 1 << 20;
  unsigned int blockSize, gridSize;

  // Host data
  TYPE *h1_a;
  TYPE *h1_b;
  TYPE *h1_c;
  TYPE *h2_a;
  TYPE *h2_b;
  TYPE *h2_c;

  // Device data
  TYPE *d1_a;
  TYPE *d1_b;
  TYPE *d1_c;
  TYPE *d2_a;
  TYPE *d2_b;
  TYPE *d2_c;

  size_t bytes = n*sizeof(TYPE);

  // Host allocation
  h1_a = (TYPE*)malloc(bytes);
  h1_b = (TYPE*)malloc(bytes);
  h1_c = (TYPE*)malloc(bytes);
  h2_a = (TYPE*)malloc(bytes);
  h2_b = (TYPE*)malloc(bytes);
  h2_c = (TYPE*)malloc(bytes);

  // Device allocation
  cudaMalloc(&d1_a, bytes);
  cudaMalloc(&d1_b, bytes);
  cudaMalloc(&d1_c, bytes);
  cudaMalloc(&d2_a, bytes);
  cudaMalloc(&d2_b, bytes);
  cudaMalloc(&d2_c, bytes);

  // Dimensions
  blockSize = 64;
  gridSize = (n + blockSize - 1)/blockSize;

  for(i=0; i<5; i++)
  {
   // MemCpy H2D of Kernel 1
   cudaMemcpy(d1_a, h1_a, bytes, cudaMemcpyHostToDevice);
   cudaMemcpy(d1_b, h1_b, bytes, cudaMemcpyHostToDevice);

   // MemCpy H2D of Kernel 2
   cudaMemcpy(d2_a, h2_a, bytes, cudaMemcpyHostToDevice);
   cudaMemcpy(d2_b, h2_b, bytes, cudaMemcpyHostToDevice);

   // Computation Kernel 1
   operation1<<<gridSize, blockSize>>>(d1_a, d1_b, d1_c, n);

   // Computation Kernel 2
   operation2<<<gridSize, blockSize>>>(d2_a, d2_b, d2_c, n);

   // MemCpy D2H of Kernel 1
   cudaMemcpy(h1_c, d1_c, bytes, cudaMemcpyDeviceToHost);

   // MemCpy D2H of Kernel 2
   cudaMemcpy(h2_c, d2_c, bytes, cudaMemcpyDeviceToHost);
  }

  cudaDeviceSynchronize();

  // Release memory
  cudaFree(h1_a);
  cudaFree(h1_b);
  cudaFree(h1_c);
  cudaFree(h2_a);
  cudaFree(h2_b);
  cudaFree(h2_c);

  return 0;
}
