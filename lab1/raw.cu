#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__ void errorexit(const char *s)
{
    printf("\n%s", s);
    exit(EXIT_FAILURE);
}

// Kernel do obliczania histogramu
__global__ void computeHistogram(int *data, int *histogram, int N, int A, int B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // Obliczenie indeksu binu dla wartości danych
        int resultIdx = (data[idx] - A);
        atomicAdd(&histogram[resultIdx], 1); // Atomiczne dodanie do histogramu
    }
}

// Kernel do obliczania sumy liczb na GPU
__global__ void computeSum(int *data, int N, int *sum)
{
    __shared__ int sharedSum[1024]; // Pamięć dzielona dla sumy w obrębie bloku

    int threadId = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadId;

    sharedSum[threadId] = (globalId < N) ? data[globalId] : 0;
    __syncthreads();

    // Suma w ramach bloku
    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        if (threadId < i)
        {
            sharedSum[threadId] += sharedSum[threadId + i];
        }
        __syncthreads();
    }

    // Wątek 0 zapisuje wynik bloku do globalnej pamięci
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

    blocksingrid = (N + threadsinblock - 1) / threadsinblock;

    printf("The kernel will run with: %d blocks\n", blocksingrid);

    int *histogramHost, *histogramDevice, *randomNumbersDevice, *sumHost, *sumDevice;

    histogramHost = (int *)calloc((B - A + 1), sizeof(int));
    sumHost = (int *)malloc(sizeof(int));
    *sumHost = 0;

    if (histogramHost == NULL || sumHost == NULL)
    {
        printf("Memory allocation failed.\n");
        return 1;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc((void **)&randomNumbersDevice, N * sizeof(int));
    cudaMalloc((void **)&histogramDevice, (B - A + 1) * sizeof(int));
    cudaMalloc((void **)&sumDevice, sizeof(int));

    cudaMemcpy(randomNumbersDevice, randomNumbers, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(histogramDevice, 0, (B - A + 1) * sizeof(int));
    cudaMemset(sumDevice, 0, sizeof(int));

    // Uruchomienie kernela histogramu
    computeHistogram<<<blocksingrid, threadsinblock>>>(randomNumbersDevice, histogramDevice, N, A, B);

    // Uruchomienie kernela sumy
    computeSum<<<blocksingrid, threadsinblock>>>(randomNumbersDevice, N, sumDevice);

    // Kopiowanie wyników z powrotem do hosta
    cudaMemcpy(histogramHost, histogramDevice, (B - A + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(sumHost, sumDevice, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);

    // Obliczenie średniej
    float average = (float)(*sumHost) / N;

    // Wydrukowanie wyników
    printf("Histogram:\n");
    for (int i = 0; i < (B - A + 1); i++)
    {
        if (histogramHost[i] > 0)
        {
            printf("%d occurs %d times\n", i + A, histogramHost[i]);
        }
    }

    printf("Sum: %d\n", *sumHost);
    printf("Average: %.2f\n", average);
    printf("Kernel execution time: %.3f ms\n", milliseconds);

    // Zwolnienie pamięci
    free(randomNumbers);
    free(histogramHost);
    free(sumHost);
    cudaFree(randomNumbersDevice);
    cudaFree(histogramDevice);
    cudaFree(sumDevice);

    return 0;
}
