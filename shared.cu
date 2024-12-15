#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__ void errorexit(const char *s)
{
    printf("\n%s", s);
    exit(EXIT_FAILURE);
}

__global__ void computeSumAndAverageShared(int *data, int N, int *sum)
{
    // Deklaracja pamięci dzielonej
    extern __shared__ int sharedSum[];

    int threadId = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadId;

    // Inicjalizowanie pamięci dzielonej
    if (threadId == 0)
    {
        sharedSum[0] = 0;
    }
    __syncthreads();

    // Jeśli wątek jest w zakresie danych
    if (globalId < N)
    {
        atomicAdd(&sharedSum[0], data[globalId]); // Dodaj do sumy w pamięci dzielonej
    }
    __syncthreads();

    // Przepisanie sumy do globalnej pamięci
    if (threadId == 0)
    {
        atomicAdd(sum, sharedSum[0]);
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

    printf("Enter the number of elements: \n");
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

    blocksingrid = (N + threadsinblock - 1) / threadsinblock;

    printf("The kernel will run with: %d blocks\n", blocksingrid);

    int *sumHost, *sumDevice;
    sumHost = (int *)malloc(sizeof(int));
    *sumHost = 0;

    int *randomNumbersDevice;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc((void **)&randomNumbersDevice, N * sizeof(int));
    cudaMalloc((void **)&sumDevice, sizeof(int));

    cudaMemcpy(randomNumbersDevice, randomNumbers, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(sumDevice, 0, sizeof(int)); // Inicjalizuj sumę na GPU

    // Rozmiar pamięci dzielonej
    int sharedMemorySize = sizeof(int);

    computeSumAndAverageShared<<<blocksingrid, threadsinblock, sharedMemorySize>>>(randomNumbersDevice, N, sumDevice);

    // Kopiuj wynik sumy do hosta
    cudaMemcpy(sumHost, sumDevice, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Oblicz średnią
    float average = (float)(*sumHost) / N;

    printf("Sum: %d\n", *sumHost);
    printf("Average: %.2f\n", average);
    printf("Kernel execution time: %.3f ms\n", milliseconds);

    // Free allocated memory
    free(randomNumbers);
    free(sumHost);
    cudaFree(randomNumbersDevice);
    cudaFree(sumDevice);

    return 0;
}
