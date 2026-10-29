// =====================================================================
// verify.hpp - Check that the GPU produced the correct results
// =====================================================================
// Stage-isolated verification: every GPU stage is checked against the
// CPU running the SAME stage on the SAME input (the GPU's previous output).
// That way a tiny rounding difference in one stage cannot snowball into
// "failures" in later stages, and a real bug points to the exact stage.
// =====================================================================
#pragma once

#include "common.hpp"

#include <cstddef>

struct CompareResult {
    int maxDiff = 0;           // largest per-pixel difference found
    size_t mismatches = 0;     // pixels that differ at all
    size_t total = 0;          // pixels compared
    int tolerance = 0;         // maxDiff allowed to pass
    bool sizeOk = false;
    bool pass = false;
};

CompareResult compareImages(const Image& a, const Image& b, int tolerance);

struct VerifyReport {
    static constexpr int kCount = 4;
    const char* names[kCount] = {"grayscale", "equalized", "gaussian blur", "sobel edges"};
    CompareResult results[kCount];

    bool allPass() const {
        for (const auto& r : results) if (!r.pass) return false;
        return true;
    }
};

// Blur gets tolerance 1: GPU and CPU may round a float sum differently
// (the GPU uses fused multiply-add). All other stages must match exactly.
VerifyReport verifyStages(const Image& rgb, const PipelineOutputs& gpu,
                          const std::vector<float>& weights);
