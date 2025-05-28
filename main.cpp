#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <filesystem>  // C++17 filesystem library
#include <vector>
#include <cmath>
#include <numeric>
namespace fs = std::filesystem;

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

void saveMapToJSON(const std::unordered_map<int, std::chrono::duration<double>>& myMap, const std::string& filename) {
    nlohmann::json j_map;
    for (const auto& pair : myMap) {
        j_map[std::to_string(pair.first)] = pair.second.count(); // duration in seconds
    }

    std::ofstream file(filename);
    if (file.is_open()) {
        file << j_map.dump(4); // Pretty-print with 4 spaces indentation
        file.close();
        std::cout << "Map saved to " << filename << " successfully." << std::endl;
    } else {
        std::cerr << "Could not open file for writing." << std::endl;
    }
}

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

bool saveImage(const std::string &filename, const unsigned char* image, int width, int height, int channels) {
    if (!stbi_write_png(filename.c_str(), width, height, channels, image, width * channels)) {
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

// Separable convolution
void separableConvolution(const unsigned char* input, unsigned char* output,
                          int width, int height,
                          const float* kernel, int kernelRadius) {
    float* temp = new float[width * height]();

    // Horizontal pass
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            for (int k = -kernelRadius; k <= kernelRadius; k++) {
                int xx = std::min(std::max(x + k, 0), width - 1);
                sum += kernel[kernelRadius + k] * input[y * width + xx];
            }
            temp[y * width + x] = sum;
        }
    }

    // Vertical pass
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            for (int k = -kernelRadius; k <= kernelRadius; k++) {
                int yy = std::min(std::max(y + k, 0), height - 1);
                sum += kernel[kernelRadius + k] * temp[yy * width + x];
            }
            int val = static_cast<int>(sum);
            output[y * width + x] = static_cast<unsigned char>(std::min(std::max(val, 0), 255));
        }
    }
    delete[] temp;
}

// Non-separable convolution
void nonSeparableConvolution(const unsigned char* input, unsigned char* output,
                             int width, int height,
                             const float* kernel, int kernelSize) {
    int kernelRadius = kernelSize / 2;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
                for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
                    int yy = std::min(std::max(y + ky, 0), height - 1);
                    int xx = std::min(std::max(x + kx, 0), width - 1);
                    sum += kernel[(ky + kernelRadius) * kernelSize + (kx + kernelRadius)] * input[yy * width + xx];
                }
            }
            int val = static_cast<int>(sum);
            output[y * width + x] = static_cast<unsigned char>(std::min(std::max(val, 0), 255));
        }
    }
}

int main() {
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
    int numKernels = 3;
    int kernelRadii[numImages] = {1, 4, 7}; // Radii for the Gaussian kernel
    int kernelWidths[numImages] = {3, 9, 15}; // Widths for the Laplacian of Gaussian kernel

    auto separableKernels = new float*[numImages];
    auto nonSeparableKernels = new float*[numImages];
    separableKernels[0] = gaussian1D(kernelRadii[0], (float(kernelWidths[0] -1))/6); // 3x3 Gaussian kernel
    separableKernels[1] = gaussian1D(kernelRadii[1], (float(kernelWidths[1] -1))/6); // 9x9 Gaussian kernel
    separableKernels[2] = gaussian1D(kernelRadii[2], (float(kernelWidths[2] -1))/6); // 15x15 Gaussian kernel
    nonSeparableKernels[0] = logKernel(kernelRadii[0], (float(kernelWidths[0] -1))/6); // 3x3 Laplacian of Gaussian kernel
    nonSeparableKernels[1] = logKernel(kernelRadii[1], (float(kernelWidths[1] -1))/6); // 9x9 Laplacian of Gaussian kernel
    nonSeparableKernels[2] = logKernel(kernelRadii[2], (float(kernelWidths[2] -1))/6); // 15x15 Laplacian of Gaussian kernel



    const int repetitions = 1; //TODO set to 100 for performance testing
    std::unordered_map<int, std::chrono::duration<double>> kernelTimes;
    std::string outDir = "output";
    if (!fs::exists(outDir)) {
        fs::create_directory(outDir);
    }

    std::chrono::duration<double> totalTime(0);

    for (size_t i = 0; i < numImages; i++) {
        int w = widths[i];
        int h = heights[i];
        unsigned char* output = new unsigned char[w * h]();
        for (int rep = 0; rep < repetitions; rep++) {
            auto start = std::chrono::high_resolution_clock::now();
            separableConvolution(images[i], output, w, h, separableKernels[i], kernelRadii[i]);
            auto end = std::chrono::high_resolution_clock::now();
            totalTime += (end - start);
            if (rep == 0) {
                std::string outFilename = outDir + "/separable_kernel_" + std::to_string(i) + ".png";
                if (saveImage(outFilename, output, w, h, 1))
                    std::cout << "Saved output image: " << outFilename << std::endl;
            }

        }
        double avgTime = totalTime.count() / (repetitions);
        kernelTimes[2*i] = std::chrono::duration<double>(avgTime);
        std::cout << "Separable Kernel average processing time - " << resDict[i] << ": " << avgTime << " seconds." << std::endl;

        for (int rep = 0; rep < repetitions; rep++) {
            totalTime = std::chrono::duration<double>(0);
            auto start = std::chrono::high_resolution_clock::now();
            nonSeparableConvolution(images[i], output, w, h, nonSeparableKernels[i], kernelWidths[i]);
            auto end = std::chrono::high_resolution_clock::now();
            totalTime += (end - start);
            if (rep == 0) {
                std::string outFilename = outDir + "/non_separable_kernel_" + std::to_string(i) + ".png";
                if (saveImage(outFilename, output, w, h, 1))
                    std::cout << "Saved output image: " << outFilename << std::endl;
            }

        }
        avgTime = totalTime.count() / (repetitions);
        kernelTimes[2*i + 1] = std::chrono::duration<double>(avgTime);
        std::cout << "Non-separable Kernel average processing time -  " << resDict[i] << ": " << avgTime << " seconds." << std::endl;
        delete[] output;
    }


    {

        for (int rep = 0; rep < repetitions; rep++) {
            for (size_t i = 0; i < numImages; i++) {

            }
        }

    }

    saveMapToJSON(kernelTimes, "sequential_results.json");

    for (size_t i = 0; i < numImages; i++) {
        delete[] images[i];
    }
    delete[] images;
    delete[] widths;
    delete[] heights;
    delete[] channelsList;
    delete[] separableKernels;
    delete[] nonSeparableKernels;

    return 0;
}
