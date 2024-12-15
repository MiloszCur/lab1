/*
CUDA - prepare the histogram of N numbers in range of <a;b> where a and b should be integers
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__ void errorexit(const char *s)
{
    printf("\n%s", s);
    exit(EXIT_FAILURE);
}

__global__ void computeHistogramSharedMemory(int *data, int *globalHistogram, int N, int A, int B)
{
    // Declare shared memory for the histogram
    extern __shared__ int sharedHistogram[];

    // Initialize shared memory histogram to 0
    int threadId = threadIdx.x;
    if (threadId < (B - A))
    {
        sharedHistogram[threadId] = 0;
    }
    __syncthreads();

    // Calculate global thread ID
    int globalId = blockIdx.x * blockDim.x + threadId;

    // Process data points in this block
    if (globalId < N)
    {
        int resultIdx = (data[globalId] - A);
        atomicAdd(&sharedHistogram[resultIdx], 1);
    }
    __syncthreads();

    // Write shared histogram back to global memory
    if (threadId < (B - A))
    {
        atomicAdd(&globalHistogram[threadId], sharedHistogram[threadId]);
    }
}

void generateRandomNumbers(int *arr, int N, int A, int B)
{

    srand(time(NULL));

    for (int i = 0; i < N; i++)
    {
        arr[i] = A + rand() % (B - A + 1);
    }
}

int main(int argc, char **argv)
{

    int threadsinblock = 1024;
    int blocksingrid;

    int N, A, B;
    cudaEvent_t start, stop;
    float milliseconds = 0;

    printf("Enter number of elements: \n");
    scanf("%d", &N);

    printf("Enter A value (start range): \n");
    scanf("%d", &A);

    printf("Enter B value (end range): \n");
    scanf("%d", &B);

    int *randomNumbers = (int *)malloc(N * sizeof(int));
    if (randomNumbers == NULL)
    {
        printf("Memory allocation failed.\n");
        return 1;
    }

    generateRandomNumbers(randomNumbers, N, A, B);

    blocksingrid = ceil((double)N / threadsinblock);

    printf("The kernel will run with: %d blocks\n", blocksingrid);

    int *resultArrayHost, *resultArrayDevice, *randomNumbersDevice;

    resultArrayHost = (int *)calloc((B - A), sizeof(int));

    if (resultArrayHost == NULL)
    {
        printf("Memory allocation failed.\n");
        return 1;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc((void **)&randomNumbersDevice, N * sizeof(int));
    cudaMalloc((void **)&resultArrayDevice, (B - A) * sizeof(int));

    cudaMemcpy(randomNumbersDevice, randomNumbers, N * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize device histogram to 0
    cudaMemset(resultArrayDevice, 0, (B - A) * sizeof(int));

    int sharedMemorySize = (B - A) * sizeof(int);

    computeHistogramSharedMemory<<<blocksingrid, threadsinblock, sharedMemorySize>>>(randomNumbersDevice, resultArrayDevice, N, A, B);

    // Copy the histogram result back to the host
    cudaMemcpy(resultArrayHost, resultArrayDevice, (B - A) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);

    // Wait for the stop event to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the histogram
    printf("Histogram:\n");
    for (int i = 0; i < B - A; i++)
    {
        printf("%d occures %d\n", i, resultArrayHost[i]);
    }

    printf("Kernel execution time: %.3f ms\n", milliseconds);
    // Free allocated memory
    free(randomNumbers);
    free(resultArrayHost);
    cudaFree(randomNumbersDevice);
    cudaFree(resultArrayDevice);

    return 0;
}
