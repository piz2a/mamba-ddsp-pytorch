#pragma once

#include <array>
#include <memory>
#include <span>
#include <vector>

#include <onnxruntime_cxx_api.h>

class ModelRuntime
{
public:
    struct CrepeResult
    {
        float f0Hz = 0.0f;
        float periodicity = 0.0f;
    };

    struct ControllerResult
    {
        std::array<float, 16> wavetableWeights {};
        float sustainAmplitude = 0.0f;
        std::array<float, 65> noiseBands {};
        float transientGain = 0.0f;
    };

    ModelRuntime();

    CrepeResult inferCrepe (std::array<float, 1024> frame);
    ControllerResult inferController (float pitch,
                                      float loudness,
                                      int64_t articulation,
                                      float onsetStrength,
                                      float offset,
                                      float gate,
                                      float noteAge,
                                      float periodicity);
    std::array<float, 20> inferOnsetEnvelope (int64_t articulation,
                                              float normalizedF0,
                                              float loudness);
    void reset();

private:
    static Ort::Session makeSession (Ort::Env&, const void*, std::size_t);
    static float binToHz (float bin);

    Ort::Env environment;
    Ort::Session crepeSession;
    Ort::Session controllerSession;
    Ort::Session onsetSession;
    Ort::MemoryInfo memory;
    std::array<float, 256> hiddenState {};
};
