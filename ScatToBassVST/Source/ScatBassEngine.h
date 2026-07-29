#pragma once

#include "ModelRuntime.h"

#include <array>
#include <atomic>
#include <memory>
#include <vector>

#include <aubio.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>

class ScatBassEngine : private juce::Thread
{
public:
    struct Telemetry
    {
        float inputLevel = 0.0f;
        float f0Hz = 0.0f;
        float periodicity = 0.0f;
        float onset = 0.0f;
        float offset = 0.0f;
        float gate = 0.0f;
        float noteAge = 0.0f;
        int articulation = 0;
        float inferenceMs = 0.0f;
    };

    ScatBassEngine();
    ~ScatBassEngine() override;

    void prepare (double hostSampleRate, int maximumBlockSize);
    void reset();
    void pushInput (const juce::AudioBuffer<float>& input);
    void pullOutput (juce::AudioBuffer<float>& output);
    void setStyle (int styleIndex);
    Telemetry getTelemetry() const;
    int latencySamples() const noexcept { return frameSizeAtHostRate; }

private:
    class MonoFifo;

    void run() override;
    bool renderOneFrame();
    bool processAubio (const float* hop);
    int resolveArticulation (float bassF0) const;
    int onsetModelArticulation (int bassArticulation) const;
    void updateNoteState (bool onsetCandidate,
                          bool noiseGate,
                          float periodicity);
    void synthesize (const ModelRuntime::ControllerResult& controls,
                     float f0,
                     float loudness,
                     float periodicity);
    void buildNoiseImpulse (const std::array<float, 65>& bands);
    void publishTelemetry (float inputLevel, float inferenceMs);

    static constexpr int modelSampleRate = 16000;
    static constexpr int modelFrameSize = 1024;
    static constexpr int modelHopSize = 256;
    static constexpr int aubioHopSize = 128;
    static constexpr int aubioWindowSize = 512;

    std::unique_ptr<MonoFifo> inputFifo;
    std::unique_ptr<MonoFifo> outputFifo;
    juce::WaitableEvent workAvailable;
    double hostRate = 48000.0;
    int frameSizeAtHostRate = 3072;
    int hopSizeAtHostRate = 768;
    juce::LagrangeInterpolator inputResampler;
    juce::LagrangeInterpolator outputResampler;
    std::vector<float> hostFrame;
    std::vector<float> inputScratch;
    std::array<float, modelFrameSize> modelFrame {};
    std::array<float, modelHopSize> synthesisFrame {};
    std::vector<float> hostOutputFrame;

    ModelRuntime models;
    aubio_onset_t* onsetDetector = nullptr;
    fvec_t* aubioInput = nullptr;
    fvec_t* aubioOutput = nullptr;

    std::array<float, 16 * 512> wavetables {};
    std::array<float, 6 * 4800> transientBank {};
    std::array<float, 20> onsetEnvelope {};
    std::array<float, 256> noiseKernel {};
    std::array<float, 256> noiseHistory {};
    int noiseHistoryIndex = 0;
    juce::Random random { 20260729 };
    float wavetablePhase = 0.0f;
    std::array<float, 16> previousWeights {};
    float previousAmplitude = 0.0f;

    std::atomic<int> style { 0 };
    bool noteActive = false;
    int activeArticulation = 0;
    float noteAgeSeconds = 0.0f;
    int onsetEnvelopeFrame = 20;
    int framesSinceOnset = 100000;
    int gateReleaseFrames = 0;
    int lowPeriodicityFrames = 0;
    bool currentOnset = false;
    bool currentOffset = false;
    bool currentGate = false;
    float currentF0 = 0.0f;
    float currentPeriodicity = 0.0f;
    float currentLoudness = 0.0f;

    int calibrationFrames = 0;
    float noisePeakDb = -100.0f;
    float loudnessMean = -24.0f;
    float loudnessVariance = 144.0f;

    std::atomic<float> telemetryInput { 0.0f };
    std::atomic<float> telemetryF0 { 0.0f };
    std::atomic<float> telemetryPeriodicity { 0.0f };
    std::atomic<float> telemetryOnset { 0.0f };
    std::atomic<float> telemetryOffset { 0.0f };
    std::atomic<float> telemetryGate { 0.0f };
    std::atomic<float> telemetryAge { 0.0f };
    std::atomic<int> telemetryArticulation { 0 };
    std::atomic<float> telemetryInferenceMs { 0.0f };
};
