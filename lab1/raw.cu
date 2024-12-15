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

__global__ void computeHistogram(int *data, int *histogram, int N, int A, int B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // Calculate the bin index for the data value
        int resultIdx = (data[idx] - A);
        atomicAdd(&histogram[resultIdx], 1); // Atomic update to avoid race conditions
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

// Function to calculate the average of the array
float calculateAverage(int *arr, int N)
{
    long long sum = 0; // Use long long to avoid overflow
    for (int i = 0; i < N; i++)
    {
        sum += arr[i];
    }
    return (float)sum / N;
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

    computeHistogram<<<blocksingrid, threadsinblock>>>(randomNumbersDevice, resultArrayDevice, N, A, B);

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
        printf("%d occurs %d times\n", i, resultArrayHost[i]);
    }

    printf("Kernel execution time: %.3f ms\n", milliseconds);

    // Calculate the average
    float average = calculateAverage(randomNumbers, N);
    printf("Average value: %.2f\n", average);

    // Free allocated memory
    free(randomNumbers);
    free(resultArrayHost);
    cudaFree(randomNumbersDevice);
    cudaFree(resultArrayDevice);

    return 0;
}
