#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void computeSumSharedMemory(int *data, int *result, int N)
{
    extern __shared__ int sharedMemory[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int threadId = threadIdx.x;

    // Initialize shared memory
    sharedMemory[threadId] = (idx < N) ? data[idx] : 0;
    __syncthreads();

    // Reduce in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadId < stride)
        {
            sharedMemory[threadId] += sharedMemory[threadId + stride];
        }
        __syncthreads();
    }

    // Thread 0 accumulates block result to global memory
    if (threadId == 0)
    {
        atomicAdd(result, sharedMemory[0]);
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

int main()
{
    int threadsPerBlock = 1024, N, A, B;
    int *dataHost, *dataDevice, *resultDevice, resultHost = 0;

    printf("Enter the number of elements: ");
    scanf("%d", &N);
    printf("Enter the range start (A): ");
    scanf("%d", &A);
    printf("Enter the range end (B): ");
    scanf("%d", &B);

    dataHost = (int *)malloc(N * sizeof(int));
    generateRandomNumbers(dataHost, N, A, B);

    cudaMalloc((void **)&dataDevice, N * sizeof(int));
    cudaMalloc((void **)&resultDevice, sizeof(int));

    cudaMemcpy(dataDevice, dataHost, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(resultDevice, 0, sizeof(int));

    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemorySize = threadsPerBlock * sizeof(int);
    computeSumSharedMemory<<<blocks, threadsPerBlock, sharedMemorySize>>>(dataDevice, resultDevice, N);

    cudaMemcpy(&resultHost, resultDevice, sizeof(int), cudaMemcpyDeviceToHost);

    float average = (float)resultHost / N;
    printf("Sum: %d, Average: %.2f\n", resultHost, average);

    free(dataHost);
    cudaFree(dataDevice);
    cudaFree(resultDevice);

    return 0;
}
