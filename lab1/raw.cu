#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void computeSum(int *data, int *result, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int localSum;

    // Thread 0 of each block initializes the result for this block
    if (threadIdx.x == 0)
        localSum = 0;
    __syncthreads();

    // Compute partial sum
    if (idx < N)
    {
        atomicAdd(&localSum, data[idx]);
    }
    __syncthreads();

    // Accumulate block sums into the global result
    if (threadIdx.x == 0)
        atomicAdd(result, localSum);
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
    computeSum<<<blocks, threadsPerBlock>>>(dataDevice, resultDevice, N);

    cudaMemcpy(&resultHost, resultDevice, sizeof(int), cudaMemcpyDeviceToHost);

    float average = (float)resultHost / N;
    printf("Sum: %d, Average: %.2f\n", resultHost, average);

    free(dataHost);
    cudaFree(dataDevice);
    cudaFree(resultDevice);

    return 0;
}
