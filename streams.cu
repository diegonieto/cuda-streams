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
  cudaStream_t stream1, stream2, stream3, stream4, stream5, stream6;
  cudaEvent_t e1, e2, e3, e4;

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
  cudaMallocHost(&h1_a, bytes);
  cudaMallocHost(&h1_b, bytes);
  cudaMallocHost(&h1_c, bytes);
  cudaMallocHost(&h2_a, bytes);
  cudaMallocHost(&h2_b, bytes);
  cudaMallocHost(&h2_c, bytes);

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

  // Streams
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);
  cudaStreamCreate(&stream4);
  cudaStreamCreate(&stream5);
  cudaStreamCreate(&stream6);

  // Events
  cudaEventCreate(&e1);
  cudaEventCreate(&e2);
  cudaEventCreate(&e3);
  cudaEventCreate(&e4);

  for(i=0; i<5; i++)
  {
   // MemCpy H2D of Kernel 1
   cudaMemcpyAsync(d1_a, h1_a, bytes, cudaMemcpyHostToDevice, stream1);
   cudaMemcpyAsync(d1_b, h1_b, bytes, cudaMemcpyHostToDevice, stream1);
   cudaEventRecord(e1, stream1);

   // MemCpy H2D of Kernel 2
   cudaMemcpyAsync(d2_a, h2_a, bytes, cudaMemcpyHostToDevice, stream4);
   cudaMemcpyAsync(d2_b, h2_b, bytes, cudaMemcpyHostToDevice, stream4);
   cudaEventRecord(e3, stream4);

   // Computation Kernel 1
   cudaStreamWaitEvent(stream2, e1, 0);
   operation1<<<gridSize, blockSize, 0, stream2>>>(d1_a, d1_b, d1_c, n);
   cudaEventRecord(e2, stream2);

   // Computation Kernel 2
   cudaStreamWaitEvent(stream5, e3, 0);
   operation2<<<gridSize, blockSize, 0, stream5>>>(d2_a, d2_b, d2_c, n);
   cudaEventRecord(e4, stream5);

   // MemCpy D2H of Kernel 1
   cudaStreamWaitEvent(stream3, e2, 0);
   cudaMemcpyAsync(h1_c, d1_c, bytes, cudaMemcpyDeviceToHost, stream3);

   // MemCpy D2H of Kernel 2
   cudaStreamWaitEvent(stream6, e4, 0);
   cudaMemcpyAsync(h2_c, d2_c, bytes, cudaMemcpyDeviceToHost, stream6);
  }

  cudaDeviceSynchronize();

  // Destroy events
  cudaEventDestroy(e1);
  cudaEventDestroy(e2);
  cudaEventDestroy(e3);
  cudaEventDestroy(e4);

  // Destroy streams
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaStreamDestroy(stream3);
  cudaStreamDestroy(stream4);
  cudaStreamDestroy(stream5);
  cudaStreamDestroy(stream6);

  // Release memory
  cudaFree(d1_a);
  cudaFree(d1_b);
  cudaFree(d1_c);
  cudaFree(d2_a);
  cudaFree(d2_b);
  cudaFree(d2_c);

  cudaFree(h1_a);
  cudaFree(h1_b);
  cudaFree(h1_c);
  cudaFree(h2_a);
  cudaFree(h2_b);
  cudaFree(h2_c);

  return 0;
}
