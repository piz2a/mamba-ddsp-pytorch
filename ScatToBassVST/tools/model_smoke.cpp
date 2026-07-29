#include "ModelRuntime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{
template <typename Range>
void requireFinite (const Range& values, const char* name)
{
    if (! std::all_of (values.begin(), values.end(), [] (float value) {
            return std::isfinite (value);
        }))
        throw std::runtime_error (std::string (name) + " contains NaN or Inf");
}
}

int main()
{
    try
    {
        ModelRuntime runtime;
        std::array<float, 1024> frame {};
        for (std::size_t index = 0; index < frame.size(); ++index)
            frame[index] = 0.2f
                           * std::sin (2.0f * 3.14159265358979323846f
                                       * 110.0f * static_cast<float> (index)
                                       / 16000.0f);

        constexpr int warmup = 5;
        constexpr int iterations = 50;
        double crepeMs = 0.0;
        double controllerMs = 0.0;
        double onsetMs = 0.0;
        ModelRuntime::CrepeResult pitch;

        for (int iteration = -warmup; iteration < iterations; ++iteration)
        {
            auto started = std::chrono::steady_clock::now();
            pitch = runtime.inferCrepe (frame);
            auto finished = std::chrono::steady_clock::now();
            if (iteration >= 0)
                crepeMs += std::chrono::duration<double, std::milli> (
                               finished - started)
                               .count();

            started = std::chrono::steady_clock::now();
            const auto controller = runtime.inferController (
                pitch.f0Hz * 0.5f, 0.0f, 0, 0.8f, 0.0f, 1.0f, 0.016f, 0.9f);
            finished = std::chrono::steady_clock::now();
            if (iteration >= 0)
                controllerMs += std::chrono::duration<double, std::milli> (
                                    finished - started)
                                    .count();
            requireFinite (controller.wavetableWeights, "wavetable weights");
            requireFinite (controller.noiseBands, "noise bands");
            if (! std::isfinite (controller.sustainAmplitude)
                || ! std::isfinite (controller.transientGain))
                throw std::runtime_error ("controller scalar contains NaN or Inf");

            started = std::chrono::steady_clock::now();
            const auto envelope = runtime.inferOnsetEnvelope (0, 0.3f, 0.0f);
            finished = std::chrono::steady_clock::now();
            if (iteration >= 0)
                onsetMs += std::chrono::duration<double, std::milli> (
                               finished - started)
                               .count();
            requireFinite (envelope, "onset envelope");
        }

        if (! std::isfinite (pitch.f0Hz) || ! std::isfinite (pitch.periodicity)
            || pitch.f0Hz <= 0.0f)
            throw std::runtime_error ("CREPE output is invalid");

        std::cout << "CREPE result: " << pitch.f0Hz << " Hz, periodicity "
                  << pitch.periodicity << '\n'
                  << "Mean CREPE ONNX: " << crepeMs / iterations << " ms\n"
                  << "Mean controller ONNX: " << controllerMs / iterations
                  << " ms\n"
                  << "Mean onset-envelope ONNX: " << onsetMs / iterations
                  << " ms\n"
                  << "PASS: all embedded model outputs are finite\n";
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "FAIL: " << exception.what() << '\n';
        return 1;
    }
}
