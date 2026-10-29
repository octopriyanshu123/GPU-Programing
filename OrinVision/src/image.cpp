// =====================================================================
// image.cpp - Image loading (PPM/PGM + JPG/PNG/BMP via stb_image),
//             PPM/PGM saving and a synthetic test image generator
// =====================================================================
#include "image.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>

#include "stb_image.h"   // third_party/stb: JPG / PNG / BMP decoder

namespace {

// Read the next header token, skipping whitespace and '#' comments.
// Consumes exactly one whitespace character after the token, which is
// what the PNM format requires before the binary pixel data.
bool readToken(std::istream& in, std::string& token) {
    token.clear();
    char c;
    while (in.get(c)) {
        if (token.empty() && c == '#') {          // comment until end of line
            std::string skip;
            std::getline(in, skip);
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!token.empty()) return true;
            continue;
        }
        token.push_back(c);
    }
    return !token.empty();
}

bool parsePositiveInt(const std::string& s, int& value) {
    if (s.empty() || s.size() > 9) return false;
    for (char c : s) if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    value = std::stoi(s);
    return value > 0;
}

// JPG, PNG, BMP, TGA, GIF (first frame), PSD via stb_image.
// Gray files stay 1 channel; color files become RGB (alpha is dropped).
bool loadCompressed(const std::string& path, Image& img, std::string& error) {
    int w = 0, h = 0, fileChannels = 0;
    if (!stbi_info(path.c_str(), &w, &h, &fileChannels)) {
        error = "unsupported image format (" + std::string(stbi_failure_reason()) +
                "): " + path + "  - use JPG, PNG, BMP, TGA, GIF, PPM or PGM";
        return false;
    }
    if (size_t(w) * h > size_t(16384) * 16384) { error = "image too large"; return false; }

    const int channels = (fileChannels <= 2) ? 1 : 3;   // 2 = gray + alpha
    uint8_t* pixels = stbi_load(path.c_str(), &w, &h, &fileChannels, channels);
    if (!pixels) {
        error = "cannot decode " + path + ": " + stbi_failure_reason();
        return false;
    }
    Image result(w, h, channels);
    std::copy(pixels, pixels + result.bytes(), result.data.begin());
    stbi_image_free(pixels);   // memory was allocated by stb_image
    img = std::move(result);
    return true;
}

}  // namespace

namespace imageio {

bool load(const std::string& path, Image& img, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { error = "cannot open file: " + path; return false; }

    // Look at the first two bytes, not the file extension: "P5"/"P6" means
    // PNM (read below by hand), anything else goes to stb_image.
    char head[2] = {};
    in.read(head, 2);
    const bool isPnm = in.gcount() == 2 && head[0] == 'P' && (head[1] == '5' || head[1] == '6');
    if (!isPnm) {
        in.close();
        return loadCompressed(path, img, error);
    }
    in.seekg(0);

    std::string magic, ws, hs, ms;
    if (!readToken(in, magic) || (magic != "P5" && magic != "P6")) {
        error = "not a binary PGM (P5) or PPM (P6) file: " + path;
        return false;
    }
    int w = 0, h = 0, maxVal = 0;
    if (!readToken(in, ws) || !readToken(in, hs) || !readToken(in, ms) ||
        !parsePositiveInt(ws, w) || !parsePositiveInt(hs, h) ||
        !parsePositiveInt(ms, maxVal)) {
        error = "invalid PNM header in " + path;
        return false;
    }
    if (maxVal > 255) { error = "only 8-bit images are supported (maxval <= 255)"; return false; }
    if (size_t(w) * h > size_t(16384) * 16384) { error = "image too large"; return false; }

    const int channels = (magic == "P6") ? 3 : 1;
    Image result(w, h, channels);
    in.read(reinterpret_cast<char*>(result.data.data()),
            static_cast<std::streamsize>(result.bytes()));
    if (static_cast<size_t>(in.gcount()) != result.bytes()) {
        error = "file is truncated: " + path;
        return false;
    }
    img = std::move(result);
    return true;
}

bool save(const std::string& path, const Image& img, std::string& error) {
    if (img.empty() || (img.channels != 1 && img.channels != 3)) {
        error = "cannot save empty or unsupported image";
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) { error = "cannot write file: " + path; return false; }
    out << (img.channels == 3 ? "P6" : "P5") << "\n"
        << img.width << " " << img.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(img.data.data()),
              static_cast<std::streamsize>(img.bytes()));
    if (!out) { error = "write failed: " + path; return false; }
    return true;
}

Image generateTestImage(int width, int height, unsigned seed) {
    Image img(width, height, 3);

    // Small deterministic random generator (same seed -> same image).
    uint32_t state = seed ? seed : 1u;
    auto rnd = [&state]() {
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    };

    struct Circle { float cx, cy, r; uint8_t col[3]; };
    std::vector<Circle> circles(12);
    const float minDim = static_cast<float>(std::min(width, height));
    for (auto& c : circles) {
        c.cx = static_cast<float>(rnd() % width);
        c.cy = static_cast<float>(rnd() % height);
        c.r = (0.03f + (rnd() % 1000) / 1000.0f * 0.12f) * minDim;
        for (auto& v : c.col) v = static_cast<uint8_t>(rnd() % 256);
    }
    const int cell = std::max(4, width / 40);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float fx = static_cast<float>(x) / width;
            const float fy = static_cast<float>(y) / height;

            // Background: smooth color gradient (dark-ish, so equalization shows)
            int r = static_cast<int>(30 + 110 * fx);
            int g = static_cast<int>(30 + 110 * fy);
            int b = static_cast<int>(70 + 50 * std::sin(6.2831853f * fx));

            // Bottom-right: checkerboard (sharp edges for Sobel)
            if (fx > 0.6f && fy > 0.6f) {
                const bool on = ((x / cell) + (y / cell)) % 2 == 0;
                r = g = b = on ? 220 : 35;
            }

            // Colored circles
            for (const auto& c : circles) {
                const float dx = x - c.cx, dy = y - c.cy;
                if (dx * dx + dy * dy < c.r * c.r) { r = c.col[0]; g = c.col[1]; b = c.col[2]; }
            }

            // Noise (the blur stage removes this)
            const int noise = static_cast<int>(rnd() % 21) - 10;
            uint8_t* p = &img.data[(size_t(y) * width + x) * 3];
            p[0] = static_cast<uint8_t>(std::clamp(r + noise, 0, 255));
            p[1] = static_cast<uint8_t>(std::clamp(g + noise, 0, 255));
            p[2] = static_cast<uint8_t>(std::clamp(b + noise, 0, 255));
        }
    }
    return img;
}

Image grayToRgb(const Image& gray) {
    Image rgb(gray.width, gray.height, 3);
    for (size_t i = 0; i < gray.pixels(); ++i)
        rgb.data[i * 3] = rgb.data[i * 3 + 1] = rgb.data[i * 3 + 2] = gray.data[i];
    return rgb;
}

}  // namespace imageio
