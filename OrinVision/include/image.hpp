// =====================================================================
// image.hpp - Simple image container + PPM/PGM file loading and saving
// =====================================================================
// We use PPM (color, "P6") and PGM (gray, "P5") because they need no
// external libraries. Convert any photo with:
//     convert photo.jpg photo.ppm          (ImageMagick)
//     python3 scripts/convert_images.py --to-ppm photo.jpg
// =====================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;            // 1 = grayscale, 3 = RGB (interleaved RGBRGB...)
    std::vector<uint8_t> data;   // row-major, width * height * channels bytes

    Image() = default;
    Image(int w, int h, int c)
        : width(w), height(h), channels(c), data(size_t(w) * h * c, 0) {}

    size_t pixels() const { return size_t(width) * height; }
    size_t bytes() const { return data.size(); }
    bool empty() const { return data.empty(); }
};

namespace imageio {

// Load JPG, PNG, BMP, TGA, GIF, or a binary PPM (P6) / PGM (P5) file.
// The format is detected from the file contents, not the extension.
bool load(const std::string& path, Image& img, std::string& error);

// Save as PGM (1 channel) or PPM (3 channels).
bool save(const std::string& path, const Image& img, std::string& error);

// Make a synthetic RGB test image (gradients, circles, checkerboard, noise).
// Same seed -> same image, so benchmarks are repeatable.
Image generateTestImage(int width, int height, unsigned seed = 42);

// Expand a 1-channel image to 3 channels (R = G = B).
Image grayToRgb(const Image& gray);

// Simple drawing on an RGB image (used to show detected features).
// Everything outside the image is clipped.
void drawCross(Image& rgb, int x, int y, int size, uint8_t r, uint8_t g, uint8_t b);
void drawCircle(Image& rgb, int cx, int cy, int radius, uint8_t r, uint8_t g, uint8_t b);
void drawLine(Image& rgb, int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b);

}  // namespace imageio
