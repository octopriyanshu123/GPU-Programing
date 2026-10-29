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
#include <string>
#include <vector>

struct CompareResult {
    int maxDiff = 0;           // largest per-item difference found
    size_t mismatches = 0;     // items (pixels / keypoints / descriptors) that differ
    size_t total = 0;          // items compared
    int tolerance = 0;         // maxDiff allowed to pass
    bool sizeOk = false;
    bool pass = false;
};

CompareResult compareImages(const Image& a, const Image& b, int tolerance);

// Keypoint lists in ANY order (the GPU list order is random). Passes if the
// GPU found the same number of keypoints as the CPU and every stored GPU
// keypoint (x, y, score) is in the CPU list (no duplicates).
// maxDiff = |GPU count - CPU count|.
CompareResult compareKeypoints(const std::vector<Keypoint>& gpu, unsigned gpuFound,
                               const std::vector<Keypoint>& cpu);

// ORB features, item by item (same keypoint order).
// maxDiff = largest Hamming distance (differing descriptor bits);
// an orientation mismatch counts as a mismatch too.
CompareResult compareOrb(const std::vector<OrbFeature>& gpu, const std::vector<OrbFeature>& cpu);

struct VerifyItem {
    std::string name;
    const char* unit;          // "pixels", "keypoints", "descriptors"
    CompareResult result;
};

struct VerifyReport {
    std::vector<VerifyItem> items;

    bool allPass() const {
        for (const auto& i : items) if (!i.result.pass) return false;
        return true;
    }
};

// Blur gets tolerance 1: GPU and CPU may round a float sum differently
// (the GPU uses fused multiply-add). All other stages must match exactly.
// Features are checked only if enabled in `features`.
VerifyReport verifyStages(const Image& rgb, const PipelineOutputs& gpu,
                          const std::vector<float>& weights, const FeatureConfig& features);
