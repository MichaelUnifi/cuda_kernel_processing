#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
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

__constant__ float c_kernel[9];
__constant__ float c_kernel_non_separable[81];
__constant__ int cNTilesX;

__global__ void separableConvolutionKernel(const unsigned char* input, unsigned char* output,
    int width, int height, int kernelRadius, int blockWidth, int blockHeight, int sharedTileWidth, int sharedTileHeight, int enlargement)
{
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
        float temp = 0.0f;
        float sum = 0.0f;
        for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
            int sharedTileIndex = (localY + kernelRadius) * sharedTileWidth + localX + kernelRadius + kx;
            temp += c_kernel[kx + kernelRadius] * tileBuffer[sharedTileIndex];
        }
        for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
            sum += c_kernel[ky + kernelRadius] * temp;
        }
        int val = static_cast<int>(sum);
        if(val < 0) val = 0;
        if(val > 255) val = 255;
        output[globalY * width + globalX] = static_cast<unsigned char>(val);
    }
}

__global__ void nonSeparableConvolutionKernel(const unsigned char* input, unsigned char* output,
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
                sum += c_kernel_non_separable[(ky + kernelRadius) * (2 * kernelRadius + 1) + kx + kernelRadius] *
                       tileBuffer[sharedTileIndex];
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
bool loadImage(const std::string &filename, std::vector<unsigned char> &image,
               int &width, int &height, int &channels) {
    // Force loading as 1-channel grayscale
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 1);
    if (!data) {
        std::cerr << "Error loading image: " << filename << std::endl;
        return false;
    }
    image.assign(data, data + width * height * channels);
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

// -------------------------------
// Main: CUDA Benchmark and Processing
// -------------------------------
int main() {
    // Create output directory if it doesn't exist

    int numMeasurements = 100;
    fs::path outputDir("cuda_output");
    if (!fs::exists(outputDir))
        fs::create_directory(outputDir);

    std::string dataDir = "data";
    std::vector<std::string> imagePaths;
    for (const auto &entry : fs::directory_iterator(dataDir)) {
        if (entry.is_regular_file())
            imagePaths.push_back(entry.path().string());
    }
    std::sort(imagePaths.begin(), imagePaths.end());
    if (imagePaths.empty()) {
        std::cerr << "No images found in directory: " << dataDir << std::endl;
        return -1;
    }

    // One separable kernel (9-element 1D kernel)
    float separableKernel[9] = {0.0270f, 0.0650f, 0.1200f, 0.1750f, 0.2000f, 0.1750f, 0.1200f, 0.0650f, 0.0270f};

    // One non-separable kernel (9x9)
    float nonSeparableKernel[9][9] = {
        {  0,   0, -1, -1, -2, -1, -1,  0,  0},
        {  0,  -1, -3, -3, -5, -3, -3, -1,  0},
        { -1,  -3, -5, -5, -7, -5, -5, -3, -1},
        { -1,  -3, -5,  0,  0,  0, -5, -3, -1},
        { -2,  -5, -7,  0, 176,  0, -7, -5, -2},
        { -1,  -3, -5,  0,  0,  0, -5, -3, -1},
        { -1,  -3, -5, -5, -7, -5, -5, -3, -1},
        {  0,  -1, -3, -3, -5, -3, -3, -1,  0},
        {  0,   0, -1, -1, -2, -1, -1,  0,  0}
    };


    int kernelRadius = 4; // Since it's a 9x9 kernel, the radius is 4
    int kernelWidth = 9;

    std::vector<int> blockSizes = {32, 64, 128, 256, 512, 1024};

    std::unordered_map<int, double> blockTimeMap;

    //Separable Kernel benchmark
    for (int blockSize : blockSizes) {
        double totalTime = 0.0;
        int imageCount = 0;

        for (const auto &imagePath : imagePaths) {
            int width, height, channels;
            std::vector<unsigned char> hostInput;
            if (!loadImage(imagePath, hostInput, width, height, channels))
                continue;

            unsigned char* input = new unsigned char[width * height];
            for (int i = 0; i < width * height; i++) {
                input[i] = hostInput[i];
            }

            size_t numPixels = (width + 2 * kernelRadius) * (height + 2 * kernelRadius);
            size_t inputSize = numPixels * channels * sizeof(unsigned char);
            size_t outputSize = inputSize;

            unsigned char* d_input = nullptr;
            float* d_temp = nullptr;
            unsigned char* d_output = nullptr;
            cudaCheckError( cudaMalloc(&d_input, inputSize) );
            cudaCheckError( cudaMalloc(&d_output, outputSize) );

            cudaCheckError( cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice) );

            float* d_kernel = nullptr;
            int kernelSizeInBytes = kernelWidth * sizeof(float);
            cudaCheckError(cudaMemcpyToSymbol(c_kernel, separableKernel, kernelSizeInBytes));
            int blockWidth = tileDimensions[blockSize].first;
            int blockHeight = tileDimensions[blockSize].second;
            int numTilesX = std::ceil(float(width) / float(blockWidth));
            int numTilesY = std::ceil(float(height) / float(blockHeight));
            cudaCheckError(cudaMemcpyToSymbol(cNTilesX, &numTilesX, sizeof(int)) );

            int threadsPerBlock = blockSize;

            int blocksPerGrid = ((numTilesX * numTilesY) * blockSize + threadsPerBlock - 1) / threadsPerBlock;

            int sharedTileHeight = blockHeight + 2 * kernelRadius;
            int sharedTileWidth = blockWidth + 2 * kernelRadius;
            int sharedMemorySize = sharedTileHeight * sharedTileWidth * sizeof(unsigned char);
            int enlargement = std::ceil(float(sharedMemorySize) / float(blockSize));

            for (int i = 0; i < numMeasurements; i++) {
                cudaEvent_t start, stop;
                cudaCheckError( cudaEventCreate(&start) );
                cudaCheckError( cudaEventCreate(&stop) );

                cudaCheckError( cudaEventRecord(start) );

                separableConvolutionKernel<<<blocksPerGrid, threadsPerBlock, sharedMemorySize * sizeof(unsigned char)>>>(d_input, d_output,
                    width, height, kernelRadius, blockWidth, blockHeight, sharedTileWidth, sharedTileHeight, enlargement);
                cudaCheckError( cudaGetLastError() );

                cudaCheckError( cudaEventRecord(stop) );
                cudaCheckError( cudaEventSynchronize(stop) );

                float elapsedMs = 0;
                cudaCheckError( cudaEventElapsedTime(&elapsedMs, start, stop) );
                totalTime += elapsedMs / 1000.0;


                cudaCheckError( cudaEventDestroy(start) );
                cudaCheckError( cudaEventDestroy(stop) );
            }

            std::vector<unsigned char> hostOutput(numPixels * channels);
            cudaCheckError( cudaMemcpy(hostOutput.data(), d_output, outputSize, cudaMemcpyDeviceToHost) );

            if (blockSize == blockSizes[0]) {
                std::string outFilename = "cuda_output/separable_kernel_" + std::to_string(imageCount) + ".png";
                if (saveImage(outFilename, hostOutput, width, height, channels)) {
                    std::cout << "Saved output image: " << outFilename << std::endl;
                }
            }
            imageCount++;

            cudaFree(d_input);
            cudaFree(d_temp);
            cudaFree(d_output);
            cudaFree(d_kernel);
            free(input);
        }

        if (imageCount > 0) {
            blockTimeMap[blockSize] = totalTime / numMeasurements;
            std::cout << "Separable Kernel, Block size " << blockSize
                      << ": Average time:  " << totalTime / numMeasurements << " seconds." << std::endl;
        }
    }


    nlohmann::json j_map;
    for (const auto &pair : blockTimeMap) {
        j_map[std::to_string(pair.first)] = pair.second;
    }
    std::string jsonFilename = "cuda_separable_kernel_results.json";
    std::ofstream outFile(jsonFilename);
    if (outFile.is_open()) {
        outFile << j_map.dump(4);
        outFile.close();
        std::cout << "Saved benchmark results to " << jsonFilename << std::endl;
    } else {
        std::cerr << "Error saving JSON file: " << jsonFilename << std::endl;
    }

    //Non-separable Kernel benchmark
    for (int blockSize : blockSizes) {
        double totalTime = 0.0;
        int imageCount = 0;


        for (const auto &imagePath : imagePaths) {
            int width, height, channels;
            std::vector<unsigned char> hostInput;
            if (!loadImage(imagePath, hostInput, width, height, channels))
                continue;

            unsigned char* input = new unsigned char[width * height];
            for (int i = 0; i < width * height; i++) {
                input[i] = hostInput[i];
            }

            size_t numPixels = (width + 2 * kernelRadius) * (height + 2 * kernelRadius);
            size_t inputSize = numPixels * channels * sizeof(unsigned char);
            size_t outputSize = inputSize;


            unsigned char* d_input = nullptr;
            float* d_temp = nullptr;
            unsigned char* d_output = nullptr;
            cudaCheckError( cudaMalloc(&d_input, inputSize) );
            cudaCheckError( cudaMalloc(&d_output, outputSize) );

            cudaCheckError( cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice) );

            float* d_kernel = nullptr;
            int kernelSizeInBytes = kernelWidth * kernelWidth * sizeof(float);
            cudaCheckError(cudaMemcpyToSymbol(c_kernel_non_separable, nonSeparableKernel, kernelSizeInBytes));
            int blockWidth = tileDimensions[blockSize].first;
            int blockHeight = tileDimensions[blockSize].second;
            int numTilesX = std::ceil(float(width) / float(blockWidth));
            int numTilesY = std::ceil(float(height) / float(blockHeight));
            cudaCheckError(cudaMemcpyToSymbol(cNTilesX, &numTilesX, sizeof(int)) );

            int threadsPerBlock = blockSize;

            int blocksPerGrid = ((numTilesX * numTilesY) * blockSize + threadsPerBlock - 1) / threadsPerBlock;

            int sharedTileHeight = blockHeight + 2 * kernelRadius;
            int sharedTileWidth = blockWidth + 2 * kernelRadius;
            int sharedMemorySize = sharedTileHeight * sharedTileWidth * sizeof(unsigned char);
            int enlargement = std::ceil(float(sharedMemorySize) / float(blockSize));


            for (int i = 0; i < numMeasurements; i++) {
                cudaEvent_t start, stop;
                cudaCheckError( cudaEventCreate(&start) );
                cudaCheckError( cudaEventCreate(&stop) );

                cudaCheckError( cudaEventRecord(start) );

                nonSeparableConvolutionKernel<<<blocksPerGrid, threadsPerBlock, sharedMemorySize * sizeof(unsigned char)>>>(d_input, d_output,
                    width, height, kernelRadius, blockWidth, blockHeight, sharedTileWidth, sharedTileHeight, enlargement);
                cudaCheckError( cudaGetLastError() );

                cudaCheckError( cudaEventRecord(stop) );
                cudaCheckError( cudaEventSynchronize(stop) );

                float elapsedMs = 0;
                cudaCheckError( cudaEventElapsedTime(&elapsedMs, start, stop) );
                totalTime += elapsedMs / 1000.0; // convert ms to seconds

                cudaCheckError( cudaEventDestroy(start) );
                cudaCheckError( cudaEventDestroy(stop) );
            }

            std::vector<unsigned char> hostOutput(numPixels * channels);
            cudaCheckError( cudaMemcpy(hostOutput.data(), d_output, outputSize, cudaMemcpyDeviceToHost) );
            if (blockSize == blockSizes[0]) {
                std::string outFilename = "cuda_output/non_separable_kernel_" + std::to_string(imageCount) + ".png";
                if (saveImage(outFilename, hostOutput, width, height, channels)) {
                    std::cout << "Saved output image: " << outFilename << std::endl;
                }
            }
            imageCount++;

            cudaFree(d_input);
            cudaFree(d_temp);
            cudaFree(d_output);
            cudaFree(d_kernel);
            free(input);
        }

        if (imageCount > 0) {
            blockTimeMap[blockSize] = totalTime / numMeasurements;
            std::cout << "Non-separable Kernel, Block size " << blockSize
                      << ": Average time:  " << totalTime / numMeasurements << " seconds." << std::endl;
        }
    }

    j_map.clear();
    for (const auto &pair : blockTimeMap) {
        j_map[std::to_string(pair.first)] = pair.second;
    }
    jsonFilename = "cuda_non_separable_kernel_results.json";
    outFile = std::ofstream(jsonFilename);
    if (outFile.is_open()) {
        outFile << j_map.dump(4);
        outFile.close();
        std::cout << "Saved benchmark results to " << jsonFilename << std::endl;
    } else {
        std::cerr << "Error saving JSON file: " << jsonFilename << std::endl;
    }

    cudaDeviceReset();
    return 0;
}
