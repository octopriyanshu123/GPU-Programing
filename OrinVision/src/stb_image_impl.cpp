// =====================================================================
// stb_image_impl.cpp - Compiles the stb_image decoder (JPG, PNG, BMP ...)
// =====================================================================
// stb_image is a "single-header library": the header holds both the
// declarations and the code. Exactly ONE .cpp file defines
// STB_IMAGE_IMPLEMENTATION before including it, so the code is compiled
// once. This file does only that (warnings are disabled for it in CMake,
// because it is third-party code we do not edit).
// =====================================================================
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR       // we only need 8-bit images
#define STBI_NO_LINEAR
#define STBI_FAILURE_USERMSG
#include "stb_image.h"
