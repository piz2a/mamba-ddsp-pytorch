#include "ScatBassEngine.h"

#include <algorithm>
#include <cmath>
#include <iostream>

int main()
{
    constexpr int sampleRate = 16000;
    constexpr int blockSize = 256;
    constexpr int blocks = 150;
    ScatBassEngine engine;
    engine.prepare (sampleRate, blockSize);
    engine.setStyle (0);
    engine.setNoiseGateThresholdDb (-45.0f);
    engine.setOctaveShift (0);

    juce::AudioBuffer<float> block (1, blockSize);
    double outputEnergy = 0.0;
    int outputSamples = 0;
    bool observedGate = false;
    bool observedOnset = false;
    float observedF0 = 0.0f;

    for (int blockIndex = 0; blockIndex < blocks; ++blockIndex)
    {
        float* samples = block.getWritePointer (0);
        for (int sample = 0; sample < blockSize; ++sample)
        {
            const int absolute = blockIndex * blockSize + sample;
            const float seconds = absolute / static_cast<float> (sampleRate);
            // Start inside the former 512 ms calibration window. This guards
            // against reintroducing a gate that learns the vocal as noise.
            const bool sounding = seconds >= 0.40f && seconds < 1.60f;
            const float age = seconds - 0.40f;
            const float attack = std::clamp (age / 0.003f, 0.0f, 1.0f);
            const float click = age >= 0.0f && age < 0.003f
                                    ? 0.35f * (1.0f - age / 0.003f)
                                    : 0.0f;
            samples[sample] =
                sounding
                    ? attack * 0.25f
                          * std::sin (2.0f * 3.14159265358979323846f
                                      * 110.0f * seconds)
                          + click
                    : 0.0f;
        }

        engine.pushInput (block);
        juce::Thread::sleep (18);
        engine.pullOutput (block);
        const auto telemetry = engine.getTelemetry();
        observedGate = observedGate || telemetry.gate > 0.5f;
        observedOnset = observedOnset || telemetry.onset > 0.5f;
        if (telemetry.periodicity > 0.5f)
            observedF0 = telemetry.f0Hz;
        if (blockIndex >= 38)
            for (int sample = 0; sample < blockSize; ++sample)
            {
                const float value = block.getSample (0, sample);
                if (! std::isfinite (value))
                {
                    std::cerr << "FAIL: engine emitted NaN or Inf\n";
                    return 1;
                }
                outputEnergy += static_cast<double> (value) * value;
                ++outputSamples;
            }
    }

    const float outputRms = static_cast<float> (
        std::sqrt (outputEnergy / std::max (outputSamples, 1)));
    engine.reset();
    if (! observedOnset || ! observedGate
        || observedF0 < 90.0f || observedF0 > 130.0f
        || outputRms <= 1.0e-6f)
    {
        std::cerr << "FAIL: onset=" << observedOnset
                  << ", gate=" << observedGate
                  << ", F0=" << observedF0
                  << ", output RMS=" << outputRms << '\n';
        return 1;
    }

    std::cout << "Observed onset: yes\n"
              << "Observed active note: yes\n"
              << "Observed mapped F0: " << observedF0 << " Hz\n"
              << "Synthesized output RMS: " << outputRms << '\n'
              << "PASS: threaded vocal-to-bass engine produced finite audio\n";
    return 0;
}
