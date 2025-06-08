#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <filesystem>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;
namespace fs = std::filesystem;

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if(code != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(code)
                  << " " << file << ":" << line << std::endl;
        if(abort) exit(code);
    }
}


std::unordered_map<int, std::pair<int, int>> tileDimensions ={// dict for automatic tile dimension sizing
    {32, {8, 4}},
    {64, {8, 8}},
    {128, {16, 8}},
    {256, {16, 16}},
    {512, {32, 16}},
    {1024, {32, 32}}
};


__device__ int clamp_int(int val, int min_val, int max_val) {
    return (val < min_val) ? min_val : (val > max_val ? max_val : val);
}

__constant__ float cKernel[15]; // max width is 15, so for smaller kernels we leave some useless space for simplicity
__constant__ float cKernelNonSeparable[225]; //same for non-separable kernel, 15x15 = 225
__constant__ int cNTilesX;
__constant__ int cNTilesY;

__global__ void separableConvolutionKernelGlobal(const unsigned char* input, unsigned char* output,
    float* temp, int width, int height, float* kernel, int kernelRadius)
{
    int tx = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tx;

    if (index < width * height) {
        int x = index % width;
        int y = index / width;

        float sum = 0.0f;
        for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
            int globalX = clamp_int(x + kx, 0, width - 1);
            int globalIndex = y * width + globalX;
            sum += kernel[kx + kernelRadius] * float(input[globalIndex]);
        }
        temp[index] = sum;
        __syncthreads();
        sum = 0.0f;
        for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
            int globalY = clamp_int(y + ky, 0, height - 1);
            int tempIndex = globalY * width + x;
            sum += kernel[ky + kernelRadius] * temp[tempIndex];
        }
        output[index] = static_cast<unsigned char>(clamp_int(static_cast<int>(sum), 0, 255));
    }
}

__global__ void nonSeparableConvolutionKernelGlobal(const unsigned char* input, unsigned char* output,
    int width, int height, float* kernel, int kernelRadius, int kernelWidth) {
    int tx = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tx;

    if (index < width * height) {
        int x = index % width;
        int y = index / width;

        float sum = 0.0f;
        for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
            for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
                int globalX = clamp_int(x + kx, 0, width - 1);
                int globalY = clamp_int(y + ky, 0, height - 1);
                int globalIndex = globalY * width + globalX;
                sum += kernel[(ky + kernelRadius) * kernelWidth + (kx + kernelRadius)] * float(input[globalIndex]);
            }
        }
        output[index] = static_cast<unsigned char>(clamp_int(static_cast<int>(sum), 0, 255));
    }
}

__global__ void separableConvolutionKernelConstant(const unsigned char* input, unsigned char* output,
    float* temp, int width, int height, int kernelRadius)
{
    int tx = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tx;

    if (index < width * height) {
        int x = index % width;
        int y = index / width;

        float sum = 0.0f;
        for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
            int globalX = clamp_int(x + kx, 0, width - 1);
            int globalIndex = y * width + globalX;
            sum += cKernel[kx + kernelRadius] * float(input[globalIndex]);
        }
        temp[index] = sum;
        __syncthreads();
        sum = 0.0f;
        for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
            int globalY = clamp_int(y + ky, 0, height - 1);
            int tempIndex = globalY * width + x;
            sum += cKernel[ky + kernelRadius] * temp[tempIndex];
        }
        int val = static_cast<int>(sum);
        if(val < 0) val = 0;
        if(val > 255) val = 255;
        output[index] = static_cast<unsigned char>(val);
    }
}

__global__ void nonSeparableConvolutionKernelConstant(const unsigned char* input, unsigned char* output,
    int width, int height, int kernelRadius, int kernelWidth) {
    int tx = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tx;

    if (index < width * height) {
        int x = index % width;
        int y = index / width;

        float sum = 0.0f;
        for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
            for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
                int globalX = clamp_int(x + kx, 0, width - 1);
                int globalY = clamp_int(y + ky, 0, height - 1);
                int globalIndex = globalY * width + globalX;
                sum += cKernel[(ky + kernelRadius) * kernelWidth + (kx + kernelRadius)] * float(input[globalIndex]);
            }
        }
        output[index] = static_cast<unsigned char>(clamp_int(static_cast<int>(sum), 0, 255));
    }
}

__global__ void separableConvolutionKernelConstAndShared(const unsigned char* input, unsigned char* output,
    int width, int height, int kernelRadius, int blockWidth, int blockHeight, int sharedTileWidth, int sharedTileHeight, int enlargementFromTile, int enlargementFromTemp)
{
    int tx = threadIdx.x;

    int tileX = blockIdx.x % cNTilesX;
    int tileY = blockIdx.x / cNTilesX;
    int tileOriginX = tileX * blockWidth;
    int tileOriginY = tileY * blockHeight;

    int totalSharedPixelsTile = sharedTileWidth * sharedTileHeight;

    extern __shared__ unsigned char sharedBuffer[];
    float* tempBuffer = reinterpret_cast<float*>(sharedBuffer + totalSharedPixelsTile * sizeof(unsigned char));

    for (int i = 0; i < enlargementFromTile; i++) {
        int linearIndex = tx + i * blockDim.x;
        if (linearIndex < totalSharedPixelsTile) {
            int x = linearIndex % sharedTileWidth;
            int y = linearIndex / sharedTileWidth;
            int globalX = tileOriginX - kernelRadius + x;
            int globalY = tileOriginY - kernelRadius + y;
            int globalIndex = globalY * width + globalX;
            if (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height) {
                sharedBuffer[linearIndex] = input[globalIndex];
            } else {
                int clampedGlobalX = clamp_int(globalX, 0, width - 1);
                int clampedGlobalY = clamp_int(globalY, 0, height - 1);
                int clampedGlobalIndex = clampedGlobalY * width + clampedGlobalX;
                sharedBuffer[linearIndex] = input[clampedGlobalIndex];
            }
        }
    }
    __syncthreads();
    for (int i = 0; i < enlargementFromTemp; i++) {
        int linearIndex = tx + i * blockDim.x;
        if (linearIndex < sharedTileHeight * blockWidth) {
            int x = linearIndex % blockWidth;
            int y = linearIndex / blockWidth;
            tempBuffer[linearIndex] = 0.0f;
            int sharedTileIndex = y * sharedTileWidth + x + kernelRadius;
            for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
                if (sharedTileIndex >= 0 && sharedTileIndex < totalSharedPixelsTile) {
                    tempBuffer[linearIndex] += cKernel[kx + kernelRadius] * float(sharedBuffer[sharedTileIndex + kx]);
                }
            }
        }
    }
    __syncthreads();

    int localX = tx % blockWidth;
    int localY = tx / blockWidth;
    int globalX = tileOriginX + localX;
    int globalY = tileOriginY + localY;

    if (globalX < width && globalY < height) {
        float sum = 0.0f;
        for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
            int tempIndex = (localY + kernelRadius + ky) * blockWidth + localX;
            sum += cKernel[ky + kernelRadius] * tempBuffer[tempIndex];
        }
        int val = static_cast<int>(sum);
        if(val < 0) val = 0;
        if(val > 255) val = 255;
        output[globalY * width + globalX] = static_cast<unsigned char>(val);
    }
}

__global__ void nonSeparableConvolutionKernelConstAndShared(const unsigned char* input, unsigned char* output,
    int width, int height, int kernelRadius, int blockWidth, int blockHeight, int sharedTileWidth, int sharedTileHeight, int enlargement) {

    int tx = threadIdx.x;

    int tileX = blockIdx.x % cNTilesX;
    int tileY = blockIdx.x / cNTilesX;
    int tileOriginX = tileX * blockWidth;
    int tileOriginY = tileY * blockHeight;

    int totalSharedPixels = sharedTileWidth * sharedTileHeight;

    extern __shared__ unsigned char tileBuffer[];

    for (int i = 0; i < enlargement; i++) {
        int linearIndex = tx + i * blockDim.x;
        if (linearIndex < totalSharedPixels) {
            int x = linearIndex % sharedTileWidth;
            int y = linearIndex / sharedTileWidth;
            int globalX = tileOriginX - kernelRadius + x;
            int globalY = tileOriginY - kernelRadius + y;
            int globalIndex = globalY * width + globalX;
            if (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height) {
                tileBuffer[linearIndex] = input[globalIndex];

            } else {
                int clampedGlobalX = clamp_int(globalX, 0, width - 1);
                int clampedGlobalY = clamp_int(globalY, 0, height - 1);
                int clampedGlobalIndex = clampedGlobalY * width + clampedGlobalX;
                tileBuffer[linearIndex] = input[clampedGlobalIndex];
            }
        }
    }
    __syncthreads();

    int localX = tx % blockWidth;
    int localY = tx / blockWidth;
    int globalX = tileOriginX + localX;
    int globalY = tileOriginY + localY;

    if (globalX < width && globalY < height) {
        float sum = 0.0f;
        for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
            for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
                int sharedTileIndex = (localY + kernelRadius + ky) * sharedTileWidth + localX + kernelRadius + kx;
                sum += cKernelNonSeparable[(ky + kernelRadius) * (2 * kernelRadius + 1) + kx + kernelRadius] *
                       float(tileBuffer[sharedTileIndex]);
            }
        }
        int val = static_cast<int>(sum);
        if(val < 0) val = 0;
        if(val > 255) val = 255;
        output[globalY * width + globalX] = static_cast<unsigned char>(val);
    }
}


// -------------------------------
// Host Functions for Image I/O
// -------------------------------
bool loadImage(const std::string &filename, unsigned char* &image, int &width, int &height, int &channels) {
    // Force loading with 1 channel
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 1);
    if (!data) {
        std::cerr << "Error loading image: " << filename << std::endl;
        return false;
    }
    image = new unsigned char[width * height];
    std::copy(data, data + (width * height), image);
    stbi_image_free(data);
    return true;
}

bool saveImage(const std::string &filename, const std::vector<unsigned char> &image,
               int width, int height, int channels) {
    // Save as PNG with given channels (RGB)
    if (!stbi_write_png(filename.c_str(), width, height, channels, image.data(), width * channels)) {
        std::cerr << "Error saving image: " << filename << std::endl;
        return false;
    }
    return true;
}

// Generate a 1D Gaussian kernel of radius `r` and standard deviation `sigma`.
// Returns a pointer to a new float array of size (2*r+1). Caller must delete[].
float* gaussian1D(int r, double sigma) {
    int size = 2 * r + 1;
    float* g = new float[size];
    double sum = 0.0;
    double denom = 2.0 * sigma * sigma;
    for (int i = -r; i <= r; ++i) {
        double v = std::exp(- (i * i) / denom);
        g[i + r] = static_cast<float>(v);
        sum += v;
    }
    // normalize
    for (int i = 0; i < size; ++i) g[i] /= static_cast<float>(sum);
    return g;
}

// Generate a 2D Laplacian-of-Gaussian kernel of radius `r` and sigma `sigma`.
// Returns a pointer to a new float array of size (2*r+1)*(2*r+1), row-major. Caller must delete[].
float* logKernel(int r, double sigma) {
    int size = 2 * r + 1;
    float* k = new float[size * size];
    double denom1 = sigma * sigma;
    double denom2 = denom1 * denom1;
    double two_den2 = 2.0 * denom1;
    double sum = 0.0;

    // raw LoG
    for (int y = -r; y <= r; ++y) {
        for (int x = -r; x <= r; ++x) {
            double rsq = double(x * x + y * y);
            double norm = (rsq - two_den2) / denom2;
            double val = norm * std::exp(-rsq / two_den2);
            k[(y + r) * size + (x + r)] = static_cast<float>(val);
            sum += val;
        }
    }
    // subtract mean so sum = 0
    double mean = sum / (size * size);
    for (int j = 0; j < size; ++j)
        for (int i = 0; i < size; ++i)
            k[j * size + i] -= static_cast<float>(mean);

    return k;
}

// -------------------------------
// Main: CUDA Benchmark and Processing
// -------------------------------
int main() {
    // Create output directory if it doesn't exist
    std::unordered_map<int, std::string> resDict = {
        {0, "low resolution"},
        {1, "medium resolution"},
        {2, "high resolution"}
    };

    std::string dataDir = "data/";

    size_t numImages = 3;
    unsigned char** images = new unsigned char*[numImages];
    int* widths = new int[numImages];
    int* heights = new int[numImages];
    int* channelsList = new int[numImages];

    loadImage(dataDir + "low.jpeg", images[0], widths[0], heights[0], channelsList[0]);
    loadImage(dataDir + "medium.png", images[1], widths[1], heights[1], channelsList[1]);
    loadImage(dataDir + "high.png", images[2], widths[2], heights[2], channelsList[2]);
    std::cout << "Loaded images:" << std::endl;
    for (size_t i = 0; i < numImages; i++) {
        std::cout << "Image " << i << ": " << resDict[i] << " (" << widths[i] << "x" << heights[i] << ")" << std::endl;
    }

    int numSizes = 7;
    int numKernels = 2 * numSizes;

    auto separableKernels = new float*[numSizes];
    auto nonSeparableKernels = new float*[numSizes];
    int kernelRadii[numSizes]; // Radii for the Gaussian kernel
    int kernelWidths[numSizes]; // Widths for the Laplacian of Gaussian kernel

    const int numMeasurements = 100;
    fs::path outputDir("cuda_output");
    if (!fs::exists(outputDir))
        fs::create_directory(outputDir);

    // Define kernels from radius 1 to 7, step of 1 (7 kernels total)
    // For each radius, we will create a Gaussian kernel and a Laplacian of Gaussian kernel
    for (int i = 0; i < numSizes; i++) {
        kernelRadii[i] = i + 1;
        kernelWidths[i] = 2 * kernelRadii[i] + 1;
        separableKernels[i] = gaussian1D(kernelRadii[i], static_cast<double>(kernelWidths[i] - 1) / 6); // 3x3, 5x5, ..., 13x13 Gaussian kernels
        nonSeparableKernels[i] = logKernel(kernelRadii[i], static_cast<double>(kernelWidths[i] - 1) / 6); // 3x3, 5x5, ..., 13x13 Laplacian of Gaussian kernels
    }

    std::vector<int> blockSizes = {32, 64, 128, 256, 512, 1024};

    std::unordered_map<int, double> blockTimeMap;

    //Separable Kernel benchmark
    for (int blockSize : blockSizes) {
        for (size_t i = 0; i < numImages; i++) {

            int width = widths[i];
            int height = heights[i];

            unsigned char* input = images[i];

            size_t numPixels = width * height;
            size_t inputSize = numPixels * sizeof(unsigned char);
            size_t outputSize = inputSize;

            unsigned char* d_input = nullptr;
            float* d_temp = nullptr;
            unsigned char* d_output = nullptr;
            cudaCheckError( cudaMalloc(&d_input, inputSize) );
            cudaCheckError( cudaMalloc(&d_temp, numPixels * sizeof(float)) );
            cudaCheckError( cudaMalloc(&d_output, outputSize) );
            cudaCheckError( cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice) );
            for (int j = 0; j < numSizes; j++) {

                float* d_kernel = nullptr;
                cudaCheckError( cudaMalloc(&d_kernel, kernelWidths[j] * sizeof(float)) );
                cudaCheckError( cudaMemcpy(d_kernel, separableKernels[j], kernelWidths[j] * sizeof(float), cudaMemcpyHostToDevice) );
                int kernelSizeInBytes = kernelWidths[j] * sizeof(float);
                cudaCheckError(cudaMemcpyToSymbol(cKernel, separableKernels[j], kernelSizeInBytes));
                int blockWidth = tileDimensions[blockSize].first;
                int blockHeight = tileDimensions[blockSize].second;
                int numTilesX = std::ceil(float(width) / float(blockWidth));
                int numTilesY = std::ceil(float(height) / float(blockHeight));
                cudaCheckError(cudaMemcpyToSymbol(cNTilesX, &numTilesX, sizeof(int)) );
                cudaCheckError(cudaMemcpyToSymbol(cNTilesY, &numTilesY, sizeof(int)) );

                int threadsPerBlock = blockSize;

                int blocksPerGrid = ((numTilesX * numTilesY) * blockSize + threadsPerBlock - 1) / threadsPerBlock;

                int sharedTileHeight = blockHeight + 2 * kernelRadii[j];
                int sharedTileWidth = blockWidth + 2 * kernelRadii[j];
                int sharedTileSize = sharedTileHeight * sharedTileWidth;
                int offset = sharedTileSize % sizeof(float);
                int sharedTempSize = sharedTileHeight * blockWidth;
                int enlargementFromTile = std::ceil(float(sharedTileSize) / float(blockSize));
                int enlargementFromTemp = std::ceil(float(sharedTempSize) / float(blockSize));


                //-------- GLOBAL MEMORY BENCHMARK--------
                std::cout << "GLOBAL MEMORY BENCHMARK" << std::endl;
                double totalTime = 0.0;
                for (int k = 0; k < numMeasurements; k++) {
                    cudaEvent_t start, stop;
                    cudaCheckError( cudaEventCreate(&start) );
                    cudaCheckError( cudaEventCreate(&stop) );

                    cudaCheckError( cudaEventRecord(start) );

                    separableConvolutionKernelGlobal<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_temp,
                        width, height, d_kernel, kernelRadii[j]);
                    cudaCheckError( cudaGetLastError() );

                    cudaCheckError( cudaEventRecord(stop) );
                    cudaCheckError( cudaEventSynchronize(stop) );

                    float elapsedMs = 0;
                    cudaCheckError( cudaEventElapsedTime(&elapsedMs, start, stop) );
                    totalTime += elapsedMs / 1000.0;


                    cudaCheckError( cudaEventDestroy(start) );
                    cudaCheckError( cudaEventDestroy(stop) );
                }

                std::vector<unsigned char> hostOutput(numPixels);
                cudaCheckError( cudaMemcpy(hostOutput.data(), d_output, outputSize, cudaMemcpyDeviceToHost) );

                if (blockSize == blockSizes[0] && j == 2 * i + 2) {
                    std::string outFilename = "cuda_output/separable_kernel_" + std::to_string(i) + ".png";
                    if (saveImage(outFilename, hostOutput, width, height, 1)) {
                        std::cout << "Saved output image: " << outFilename << std::endl;
                    }
                }

                blockTimeMap[numKernels * i + j] = totalTime / numMeasurements;
                std::cout << "Separable Kernel size " << kernelWidths[j] << "x" << kernelWidths[j] <<", Block size " << blockSize << "- " << resDict[i]
                          << ": Average time:  " << totalTime / numMeasurements << " seconds." << std::endl;

                totalTime = 0.0;

                //Non-separable benchmark

                d_kernel = nullptr;
                cudaCheckError( cudaMalloc(&d_kernel, kernelWidths[j] * kernelWidths[j] * sizeof(float)) );
                cudaCheckError( cudaMemcpy(d_kernel, nonSeparableKernels[j], kernelWidths[j] * kernelWidths[j] * sizeof(float), cudaMemcpyHostToDevice) );

                for (int k = 0; k < numMeasurements; k++) {
                    cudaEvent_t start, stop;
                    cudaCheckError( cudaEventCreate(&start) );
                    cudaCheckError( cudaEventCreate(&stop) );

                    cudaCheckError( cudaEventRecord(start) );

                    nonSeparableConvolutionKernelGlobal<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output,
                        width, height, d_kernel, kernelRadii[j], kernelWidths[j]);
                    cudaCheckError( cudaGetLastError() );

                    cudaCheckError( cudaEventRecord(stop) );
                    cudaCheckError( cudaEventSynchronize(stop) );

                    float elapsedMs = 0;
                    cudaCheckError( cudaEventElapsedTime(&elapsedMs, start, stop) );
                    totalTime += elapsedMs / 1000.0;


                    cudaCheckError( cudaEventDestroy(start) );
                    cudaCheckError( cudaEventDestroy(stop) );
                }

                cudaCheckError( cudaMemcpy(hostOutput.data(), d_output, outputSize, cudaMemcpyDeviceToHost) );

                if (blockSize == blockSizes[0] && j == 2 * i + 2) {
                    std::string outFilename = "cuda_output/non_separable_kernel_" + std::to_string(i) + ".png";
                    if (saveImage(outFilename, hostOutput, width, height, 1)) {
                        std::cout << "Saved output image: " << outFilename << std::endl;
                    }
                }

                blockTimeMap[numKernels * i + j + numSizes] = totalTime / numMeasurements;
                std::cout << "Nonseparable Kernel size " << kernelWidths[j] << "x" << kernelWidths[j] << ", Block size " << blockSize << "- " << resDict[i]
                          << ": Average time:  " << totalTime / numMeasurements << " seconds." << std::endl;
                totalTime = 0.0;

                std::cout << "CONSTANT MEMORY BENCHMARK" << std::endl;
                //-------- CONSTANT MEMORY BENCHMARK--------
                for (int k = 0; k < numMeasurements; k++) {
                    cudaEvent_t start, stop;
                    cudaCheckError( cudaEventCreate(&start) );
                    cudaCheckError( cudaEventCreate(&stop) );

                    cudaCheckError( cudaEventRecord(start) );

                    separableConvolutionKernelConstant<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_temp,
                        width, height, kernelRadii[j]);
                    cudaCheckError( cudaGetLastError() );

                    cudaCheckError( cudaEventRecord(stop) );
                    cudaCheckError( cudaEventSynchronize(stop) );

                    float elapsedMs = 0;
                    cudaCheckError( cudaEventElapsedTime(&elapsedMs, start, stop) );
                    totalTime += elapsedMs / 1000.0;


                    cudaCheckError( cudaEventDestroy(start) );
                    cudaCheckError( cudaEventDestroy(stop) );
                }

                blockTimeMap[42 + numKernels * i + j] = totalTime / numMeasurements;
                std::cout << "Separable Kernel size " << kernelWidths[j] << "x" << kernelWidths[j] <<", Block size " << blockSize << "- " << resDict[i]
                          << ": Average time:  " << totalTime / numMeasurements << " seconds." << std::endl;

                totalTime = 0.0;

                //Non-separable benchmark

                cudaCheckError( cudaMalloc(&d_input, inputSize) );
                cudaCheckError( cudaMalloc(&d_output, outputSize) );

                cudaCheckError( cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice) );

                kernelSizeInBytes = kernelWidths[j] * kernelWidths[j] * sizeof(float);
                cudaCheckError(cudaMemcpyToSymbol(cKernelNonSeparable, nonSeparableKernels[j], kernelSizeInBytes));

                cudaCheckError(cudaMemcpyToSymbol(cNTilesX, &numTilesX, sizeof(int)) );

                for (int k = 0; k < numMeasurements; k++) {
                    cudaEvent_t start, stop;
                    cudaCheckError( cudaEventCreate(&start) );
                    cudaCheckError( cudaEventCreate(&stop) );

                    cudaCheckError( cudaEventRecord(start) );

                    nonSeparableConvolutionKernelConstAndShared<<<blocksPerGrid, threadsPerBlock, sharedTileSize * sizeof(unsigned char)>>>(d_input, d_output,
                        width, height, kernelRadii[j], blockWidth, blockHeight, sharedTileWidth, sharedTileHeight, enlargementFromTile);
                    cudaCheckError( cudaGetLastError() );

                    cudaCheckError( cudaEventRecord(stop) );
                    cudaCheckError( cudaEventSynchronize(stop) );

                    float elapsedMs = 0;
                    cudaCheckError( cudaEventElapsedTime(&elapsedMs, start, stop) );
                    totalTime += elapsedMs / 1000.0;


                    cudaCheckError( cudaEventDestroy(start) );
                    cudaCheckError( cudaEventDestroy(stop) );
                }

                cudaCheckError( cudaMemcpy(hostOutput.data(), d_output, outputSize, cudaMemcpyDeviceToHost) );

                blockTimeMap[42 + numKernels * i + j + numSizes] = totalTime / numMeasurements;
                std::cout << "Nonseparable Kernel size " << kernelWidths[j] << "x" << kernelWidths[j] << ", Block size " << blockSize << "- " << resDict[i]
                          << ": Average time:  " << totalTime / numMeasurements << " seconds." << std::endl;
                totalTime = 0.0;

                //-------- SHARED MEMORY BENCHMARK--------
                std::cout << "SHARED MEMORY BENCHMARK" << std::endl;
                for (int k = 0; k < numMeasurements; k++) {
                    cudaEvent_t start, stop;
                    cudaCheckError( cudaEventCreate(&start) );
                    cudaCheckError( cudaEventCreate(&stop) );

                    cudaCheckError( cudaEventRecord(start) );

                    separableConvolutionKernelConstAndShared<<<blocksPerGrid, threadsPerBlock, sharedTileSize * sizeof(unsigned char) + offset + sharedTempSize * sizeof(float)>>>(d_input, d_output,
                        width, height, kernelRadii[j], blockWidth, blockHeight, sharedTileWidth, sharedTileHeight, enlargementFromTile, enlargementFromTemp);
                    cudaCheckError( cudaGetLastError() );

                    cudaCheckError( cudaEventRecord(stop) );
                    cudaCheckError( cudaEventSynchronize(stop) );

                    float elapsedMs = 0;
                    cudaCheckError( cudaEventElapsedTime(&elapsedMs, start, stop) );
                    totalTime += elapsedMs / 1000.0;


                    cudaCheckError( cudaEventDestroy(start) );
                    cudaCheckError( cudaEventDestroy(stop) );
                }

                blockTimeMap[84 + numKernels * i + j] = totalTime / numMeasurements;
                std::cout << "Separable Kernel size " << kernelWidths[j] << "x" << kernelWidths[j] <<", Block size " << blockSize << "- " << resDict[i]
                          << ": Average time:  " << totalTime / numMeasurements << " seconds." << std::endl;

                totalTime = 0.0;
                // Reset total time for the non-separable benchmark

                //Non-separable benchmark

                cudaCheckError( cudaMalloc(&d_input, inputSize) );
                cudaCheckError( cudaMalloc(&d_output, outputSize) );

                cudaCheckError( cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice) );

                kernelSizeInBytes = kernelWidths[j] * kernelWidths[j] * sizeof(float);
                cudaCheckError(cudaMemcpyToSymbol(cKernelNonSeparable, nonSeparableKernels[j], kernelSizeInBytes));

                cudaCheckError(cudaMemcpyToSymbol(cNTilesX, &numTilesX, sizeof(int)) );

                for (int k = 0; k < numMeasurements; k++) {
                    cudaEvent_t start, stop;
                    cudaCheckError( cudaEventCreate(&start) );
                    cudaCheckError( cudaEventCreate(&stop) );

                    cudaCheckError( cudaEventRecord(start) );

                    nonSeparableConvolutionKernelConstAndShared<<<blocksPerGrid, threadsPerBlock, sharedTileSize * sizeof(unsigned char)>>>(d_input, d_output,
                        width, height, kernelRadii[j], blockWidth, blockHeight, sharedTileWidth, sharedTileHeight, enlargementFromTile);
                    cudaCheckError( cudaGetLastError() );

                    cudaCheckError( cudaEventRecord(stop) );
                    cudaCheckError( cudaEventSynchronize(stop) );

                    float elapsedMs = 0;
                    cudaCheckError( cudaEventElapsedTime(&elapsedMs, start, stop) );
                    totalTime += elapsedMs / 1000.0;


                    cudaCheckError( cudaEventDestroy(start) );
                    cudaCheckError( cudaEventDestroy(stop) );
                }

                blockTimeMap[84 + numKernels * i + j + numSizes] = totalTime / numMeasurements;
                std::cout << "Nonseparable Kernel size " << kernelWidths[j] << "x" << kernelWidths[j] << ", Block size " << blockSize << "- " << resDict[i]
                          << ": Average time:  " << totalTime / numMeasurements << " seconds." << std::endl;
                cudaFree(d_kernel);
            }
            cudaFree(d_input);
            cudaFree(d_temp);
            cudaFree(d_output);
        }

        nlohmann::json j_map;
        for (const auto &pair : blockTimeMap) {
            j_map[std::to_string(pair.first)] = pair.second;
        }
        std::string jsonFilename = "cuda_" + std::to_string(blockSize) +"_results.json";
        std::ofstream outFile(jsonFilename);
        if (outFile.is_open()) {
            outFile << j_map.dump(4);
            outFile.close();
            std::cout << "Saved benchmark results to " << jsonFilename << std::endl;
        } else {
            std::cerr << "Error saving JSON file: " << jsonFilename << std::endl;
        }
    }

    return 0;
}
