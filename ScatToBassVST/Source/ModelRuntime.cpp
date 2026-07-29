#include "ModelRuntime.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <ScatModelData.h>

namespace
{
constexpr std::array<int64_t, 2> frameShape { 1, 1024 };
constexpr std::array<int64_t, 3> scalarShape { 1, 1, 1 };
constexpr std::array<int64_t, 2> articulationShape { 1, 1 };
constexpr std::array<int64_t, 3> hiddenShape { 1, 1, 256 };
constexpr std::array<int64_t, 1> onsetArticulationShape { 1 };
constexpr std::array<int64_t, 2> onsetControlShape { 1, 2 };

template <typename T, std::size_t N>
Ort::Value tensor (Ort::MemoryInfo& memory,
                   std::array<T, N>& values,
                   const int64_t* shape,
                   std::size_t dimensions)
{
    return Ort::Value::CreateTensor<T> (
        memory, values.data(), values.size(), shape, dimensions);
}
}

Ort::Session ModelRuntime::makeSession (Ort::Env& environment,
                                        const void* bytes,
                                        std::size_t size)
{
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads (1);
    options.SetInterOpNumThreads (1);
    options.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_ALL);
    return Ort::Session (environment, bytes, size, options);
}

ModelRuntime::ModelRuntime()
    : environment (ORT_LOGGING_LEVEL_WARNING, "scat-to-bass"),
      crepeSession (makeSession (environment,
                                 scat_models::torchcrepe_tiny_onnx,
                                 scat_models::torchcrepe_tiny_onnxSize)),
      controllerSession (makeSession (environment,
                                      scat_models::bass_ddsp_controller_onnx,
                                      scat_models::bass_ddsp_controller_onnxSize)),
      onsetSession (makeSession (environment,
                                 scat_models::onset_envelope_onnx,
                                 scat_models::onset_envelope_onnxSize)),
      memory (Ort::MemoryInfo::CreateCpu (OrtArenaAllocator, OrtMemTypeDefault))
{
}

void ModelRuntime::reset()
{
    hiddenState.fill (0.0f);
}

float ModelRuntime::binToHz (float bin)
{
    const float cents = 20.0f * bin + 1997.3794084376191f;
    return 10.0f * std::pow (2.0f, cents / 1200.0f);
}

ModelRuntime::CrepeResult ModelRuntime::inferCrepe (std::array<float, 1024> frame)
{
    const float mean =
        std::accumulate (frame.begin(), frame.end(), 0.0f) / frame.size();
    float variance = 0.0f;
    for (const float value : frame)
        variance += (value - mean) * (value - mean);
    const float standardDeviation =
        std::max (std::sqrt (variance / static_cast<float> (frame.size() - 1)),
                  1.0e-10f);
    for (float& value : frame)
        value = (value - mean) / standardDeviation;

    auto input = tensor (memory, frame, frameShape.data(), frameShape.size());
    const char* inputNames[] { "audio_frame" };
    const char* outputNames[] { "pitch_probabilities" };
    auto outputs = crepeSession.Run (Ort::RunOptions { nullptr },
                                     inputNames,
                                     &input,
                                     1,
                                     outputNames,
                                     1);
    const float* probabilities = outputs[0].GetTensorData<float>();
    const int minBin = std::max (0, static_cast<int> (
        std::floor ((1200.0f * std::log2 (50.0f / 10.0f)
                     - 1997.3794084376191f)
                    / 20.0f)));
    const int maxBin = std::min (359, static_cast<int> (
        std::ceil ((1200.0f * std::log2 (1000.0f / 10.0f)
                    - 1997.3794084376191f)
                   / 20.0f)));
    int peak = minBin;
    for (int index = minBin + 1; index <= maxBin; ++index)
        if (probabilities[index] > probabilities[peak])
            peak = index;

    float weightedCents = 0.0f;
    float weightSum = 0.0f;
    for (int index = std::max (minBin, peak - 4);
         index <= std::min (maxBin, peak + 4);
         ++index)
    {
        // This second sigmoid matches torchcrepe.decode.weighted_argmax,
        // whose input is already the model's sigmoid probability.
        const float weight = 1.0f / (1.0f + std::exp (-probabilities[index]));
        weightedCents += weight * (20.0f * index + 1997.3794084376191f);
        weightSum += weight;
    }
    const float cents = weightedCents / std::max (weightSum, 1.0e-8f);
    return { 10.0f * std::pow (2.0f, cents / 1200.0f), probabilities[peak] };
}

ModelRuntime::ControllerResult ModelRuntime::inferController (
    float pitch,
    float loudness,
    int64_t articulation,
    float onsetStrength,
    float offset,
    float gate,
    float noteAge,
    float periodicity)
{
    std::array<float, 1> pitchValue { pitch };
    std::array<float, 1> loudnessValue { loudness };
    std::array<int64_t, 1> articulationValue { articulation };
    std::array<float, 1> onsetValue { onsetStrength };
    std::array<float, 1> offsetValue { offset };
    std::array<float, 1> gateValue { gate };
    std::array<float, 1> ageValue { noteAge };
    std::array<float, 1> periodicityValue { periodicity };
    std::array<Ort::Value, 9> inputs {
        tensor (memory, pitchValue, scalarShape.data(), scalarShape.size()),
        tensor (memory, loudnessValue, scalarShape.data(), scalarShape.size()),
        tensor (memory,
                articulationValue,
                articulationShape.data(),
                articulationShape.size()),
        tensor (memory, onsetValue, scalarShape.data(), scalarShape.size()),
        tensor (memory, offsetValue, scalarShape.data(), scalarShape.size()),
        tensor (memory, gateValue, scalarShape.data(), scalarShape.size()),
        tensor (memory, ageValue, scalarShape.data(), scalarShape.size()),
        tensor (memory, periodicityValue, scalarShape.data(), scalarShape.size()),
        tensor (memory, hiddenState, hiddenShape.data(), hiddenShape.size())
    };
    const char* inputNames[] {
        "pitch", "loudness", "articulation", "onset_strength", "offset",
        "gate", "note_age", "periodicity", "hidden_state"
    };
    const char* outputNames[] {
        "wavetable_weights", "sustain_amplitude", "noise_bands",
        "transient_gain", "next_hidden_state"
    };
    auto outputs = controllerSession.Run (Ort::RunOptions { nullptr },
                                          inputNames,
                                          inputs.data(),
                                          inputs.size(),
                                          outputNames,
                                          5);
    ControllerResult result;
    std::copy_n (outputs[0].GetTensorData<float>(),
                 result.wavetableWeights.size(),
                 result.wavetableWeights.begin());
    result.sustainAmplitude = outputs[1].GetTensorData<float>()[0];
    std::copy_n (outputs[2].GetTensorData<float>(),
                 result.noiseBands.size(),
                 result.noiseBands.begin());
    result.transientGain = outputs[3].GetTensorData<float>()[0];
    std::copy_n (outputs[4].GetTensorData<float>(),
                 hiddenState.size(),
                 hiddenState.begin());
    return result;
}

std::array<float, 20> ModelRuntime::inferOnsetEnvelope (
    int64_t articulation,
    float normalizedF0,
    float loudness)
{
    std::array<int64_t, 1> articulationValue { articulation };
    std::array<float, 2> controlValues { normalizedF0, loudness };
    std::array<Ort::Value, 2> inputs {
        tensor (memory,
                articulationValue,
                onsetArticulationShape.data(),
                onsetArticulationShape.size()),
        tensor (memory,
                controlValues,
                onsetControlShape.data(),
                onsetControlShape.size())
    };
    const char* inputNames[] { "articulation", "controls" };
    const char* outputNames[] { "onset_envelope" };
    auto outputs = onsetSession.Run (Ort::RunOptions { nullptr },
                                     inputNames,
                                     inputs.data(),
                                     inputs.size(),
                                     outputNames,
                                     1);
    std::array<float, 20> result {};
    std::copy_n (
        outputs[0].GetTensorData<float>(), result.size(), result.begin());
    return result;
}
