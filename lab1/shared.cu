#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__ void errorexit(const char *s)
{
    printf("\n%s", s);
    exit(EXIT_FAILURE);
}

// Kernel do obliczania histogramu z pamięcią dzieloną
__global__ void computeHistogramSharedMemory(int *data, int *globalHistogram, int N, int A, int B)
{
    // Deklaracja pamięci dzielonej dla histogramu
    extern __shared__ int sharedHistogram[];

    int threadId = threadIdx.x;
    if (threadId < (B - A + 1))
    {
        sharedHistogram[threadId] = 0;
    }
    __syncthreads();

    int globalId = blockIdx.x * blockDim.x + threadId;

    // Przetwarzanie danych w obrębie bloku
    if (globalId < N)
    {
        int resultIdx = data[globalId] - A;
        atomicAdd(&sharedHistogram[resultIdx], 1); // Atomiczne dodanie do histogramu
    }
    __syncthreads();

    // Zapisz wyniki z pamięci dzielonej do pamięci globalnej
    if (threadId < (B - A + 1))
    {
        atomicAdd(&globalHistogram[threadId], sharedHistogram[threadId]);
    }
}

// Kernel do obliczania sumy liczb na GPU
__global__ void computeSumSharedMemory(int *data, int N, int *globalSum)
{
    extern __shared__ int sharedSum[];

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
        atomicAdd(globalSum, sharedSum[0]);
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
    int threadsPerBlock = 1024;
    int blocksInGrid;

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

    blocksInGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("The kernel will run with: %d blocks\n", blocksInGrid);

    int *histogramHost, *histogramDevice, *randomNumbersDevice;
    int *sumHost, *sumDevice;

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

    int sharedMemorySizeForHistogram = (B - A + 1) * sizeof(int);
    int sharedMemorySizeForSum = threadsPerBlock * sizeof(int);

    // Uruchomienie kernela do obliczania histogramu
    computeHistogramSharedMemory<<<blocksInGrid, threadsPerBlock, sharedMemorySizeForHistogram>>>(randomNumbersDevice, histogramDevice, N, A, B);

    // Uruchomienie kernela do obliczania sumy
    computeSumSharedMemory<<<blocksInGrid, threadsPerBlock, sharedMemorySizeForSum>>>(randomNumbersDevice, N, sumDevice);

    // Kopiowanie wyników do hosta
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
            printf("%d occurs %d times\n",
