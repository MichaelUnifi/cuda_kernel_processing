#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <filesystem>  // C++17 filesystem library
#include <vector>
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
    channels = 1; // Force single channel
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

    std::string dataDir = "data";
    std::vector<std::string> imagePaths;
    for (const auto& entry : fs::directory_iterator(dataDir)) {
        if (entry.is_regular_file()) {
            imagePaths.push_back(entry.path().string());
        }
    }
    std::sort(imagePaths.begin(), imagePaths.end());
    if (imagePaths.empty()) {
        std::cerr << "No images found in directory: " << dataDir << std::endl;
        return -1;
    }


    size_t numImages = imagePaths.size();
    unsigned char** images = new unsigned char*[numImages];
    int* widths = new int[numImages];
    int* heights = new int[numImages];
    int* channelsList = new int[numImages];

    for (size_t i = 0; i < numImages; i++) {
        if (loadImage(imagePaths[i], images[i], widths[i], heights[i], channelsList[i])) {
            std::cout << "Loaded image: " << imagePaths[i] << " ("
                      << widths[i] << "x" << heights[i] << ", " << channelsList[i] << " channels)" << std::endl;
        }
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

    const int repetitions = 10;
    std::unordered_map<int, std::chrono::duration<double>> kernelTimes;
    std::string outDir = "output";
    if (!fs::exists(outDir)) {
        fs::create_directory(outDir);
    }

    std::chrono::duration<double> totalTime(0);
    for (int rep = 0; rep < repetitions; rep++) {
        for (size_t i = 0; i < numImages; i++) {
            int w = widths[i];
            int h = heights[i];
            int c = channelsList[i];
            unsigned char* output = new unsigned char[w * h * c]();
            auto start = std::chrono::high_resolution_clock::now();
            separableConvolution(images[i], output, w, h, separableKernel, kernelRadius);
            auto end = std::chrono::high_resolution_clock::now();
            totalTime += (end - start);
            if (rep == 0 && i == 0) {
                std::string outFilename = outDir + "/separable_kernel_" + std::to_string(i) + ".png";
                if (saveImage(outFilename, output, w, h, 1))
                    std::cout << "Saved output image: " << outFilename << std::endl;
            }
            delete[] output;
        }
    }
    double avgTime = totalTime.count() / (repetitions);
    kernelTimes[0] = std::chrono::duration<double>(avgTime);
    std::cout << "Separable Kernel average processing time: " << avgTime << " seconds." << std::endl;

    {
        std::chrono::duration<double> totalTime(0);
        for (int rep = 0; rep < repetitions; rep++) {
            for (size_t i = 0; i < numImages; i++) {
                int w = widths[i];
                int h = heights[i];
                unsigned char* output = new unsigned char[w * h]();
                auto start = std::chrono::high_resolution_clock::now();
                nonSeparableConvolution(images[i], output, w, h, &nonSeparableKernel[0][0], kernelWidth);
                auto end = std::chrono::high_resolution_clock::now();
                totalTime += (end - start);
                if (rep == 0 && i == 0) {
                    std::string outFilename = outDir + "/non_separable_kernel_" + std::to_string(i) + ".png";
                    if (saveImage(outFilename, output, w, h, 1))
                        std::cout << "Saved output image: " << outFilename << std::endl;
                }
                delete[] output;
            }
        }
        avgTime = totalTime.count() / (repetitions);
        kernelTimes[2] = std::chrono::duration<double>(avgTime);
        std::cout << "Non-separable Kernel average processing time: " << avgTime << " seconds." << std::endl;
    }

    saveMapToJSON(kernelTimes, "sequential_results.json");

    for (size_t i = 0; i < numImages; i++) {
        delete[] images[i];
    }
    delete[] images;
    delete[] widths;
    delete[] heights;
    delete[] channelsList;

    return 0;
}
