#include "ScatBassEngine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>

#include <ScatModelData.h>

namespace
{
constexpr float pi = 3.14159265358979323846f;

float clamp (float value, float low, float high)
{
    return std::max (low, std::min (high, value));
}

float rms (const float* values, int count)
{
    double energy = 0.0;
    for (int index = 0; index < count; ++index)
        energy += static_cast<double> (values[index]) * values[index];
    return static_cast<float> (
        std::sqrt (energy / static_cast<double> (std::max (count, 1))));
}
}

class ScatBassEngine::MonoFifo
{
public:
    explicit MonoFifo (int capacity)
        : storage (static_cast<std::size_t> (capacity), 0.0f),
          fifo (capacity)
    {
    }

    void clear()
    {
        fifo.reset();
        std::fill (storage.begin(), storage.end(), 0.0f);
    }

    int ready() const { return fifo.getNumReady(); }

    void write (const float* source, int count)
    {
        const int writable = std::min (count, fifo.getFreeSpace());
        int start1 = 0, size1 = 0, start2 = 0, size2 = 0;
        fifo.prepareToWrite (writable, start1, size1, start2, size2);
        std::copy_n (source, size1, storage.data() + start1);
        std::copy_n (source + size1, size2, storage.data() + start2);
        fifo.finishedWrite (size1 + size2);
    }

    bool peekAndAdvance (float* destination, int peekCount, int advanceCount)
    {
        if (fifo.getNumReady() < peekCount)
            return false;
        int start1 = 0, size1 = 0, start2 = 0, size2 = 0;
        fifo.prepareToRead (peekCount, start1, size1, start2, size2);
        std::copy_n (storage.data() + start1,
                     size1,
                     destination);
        std::copy_n (storage.data() + start2,
                     size2,
                     destination + size1);
        fifo.finishedRead (advanceCount);
        return true;
    }

    int read (float* destination, int count)
    {
        const int readable = std::min (count, fifo.getNumReady());
        int start1 = 0, size1 = 0, start2 = 0, size2 = 0;
        fifo.prepareToRead (readable, start1, size1, start2, size2);
        std::copy_n (storage.data() + start1,
                     size1,
                     destination);
        std::copy_n (storage.data() + start2,
                     size2,
                     destination + size1);
        fifo.finishedRead (size1 + size2);
        return readable;
    }

private:
    std::vector<float> storage;
    juce::AbstractFifo fifo;
};

ScatBassEngine::ScatBassEngine()
    : juce::Thread ("Scat-to-Bass inference")
{
    static_assert (sizeof (wavetables) == scat_models::sustain_wavetables_f32Size);
    static_assert (sizeof (transientBank) == scat_models::transient_bank_f32Size);
    std::memcpy (wavetables.data(),
                 scat_models::sustain_wavetables_f32,
                 sizeof (wavetables));
    std::memcpy (transientBank.data(),
                 scat_models::transient_bank_f32,
                 sizeof (transientBank));
    onsetDetector = new_aubio_onset (
        "complex", aubioWindowSize, aubioHopSize, modelSampleRate);
    aubioInput = new_fvec (aubioHopSize);
    aubioOutput = new_fvec (1);
    aubio_onset_set_threshold (onsetDetector, 0.30f);
    aubio_onset_set_silence (onsetDetector, -45.0f);
    aubio_onset_set_minioi_ms (onsetDetector, 80.0f);
}

ScatBassEngine::~ScatBassEngine()
{
    signalThreadShouldExit();
    workAvailable.signal();
    stopThread (2000);
    if (aubioOutput != nullptr)
        del_fvec (aubioOutput);
    if (aubioInput != nullptr)
        del_fvec (aubioInput);
    if (onsetDetector != nullptr)
        del_aubio_onset (onsetDetector);
}

void ScatBassEngine::prepare (double sampleRate, int maximumBlockSize)
{
    signalThreadShouldExit();
    workAvailable.signal();
    stopThread (2000);
    hostRate = sampleRate;
    frameSizeAtHostRate =
        static_cast<int> (std::ceil (hostRate * modelFrameSize / modelSampleRate));
    hopSizeAtHostRate =
        static_cast<int> (std::round (hostRate * modelHopSize / modelSampleRate));
    const int capacity = std::max (frameSizeAtHostRate * 16,
                                   maximumBlockSize * 32);
    inputFifo = std::make_unique<MonoFifo> (capacity);
    outputFifo = std::make_unique<MonoFifo> (capacity);
    hostFrame.assign (static_cast<std::size_t> (frameSizeAtHostRate), 0.0f);
    inputScratch.assign (
        static_cast<std::size_t> (std::max (maximumBlockSize, 1)), 0.0f);
    hostOutputFrame.assign (
        static_cast<std::size_t> (hopSizeAtHostRate), 0.0f);
    reset();
    startThread (juce::Thread::Priority::high);
}

void ScatBassEngine::reset()
{
    if (inputFifo)
        inputFifo->clear();
    if (outputFifo)
        outputFifo->clear();
    inputResampler.reset();
    outputResampler.reset();
    models.reset();
    aubio_onset_reset (onsetDetector);
    modelFrame.fill (0.0f);
    synthesisFrame.fill (0.0f);
    noiseHistory.fill (0.0f);
    noiseHistoryIndex = 0;
    wavetablePhase = 0.0f;
    previousWeights.fill (0.0f);
    previousAmplitude = 0.0f;
    noteActive = false;
    activeArticulation = 0;
    noteAgeSeconds = 0.0f;
    onsetEnvelopeFrame = static_cast<int> (onsetEnvelope.size());
    framesSinceOnset = 100000;
    gateReleaseFrames = 0;
    calibrationFrames = 0;
    noisePeakDb = -100.0f;
}

void ScatBassEngine::setStyle (int styleIndex)
{
    style.store (juce::jlimit (0, 6, styleIndex), std::memory_order_release);
}

void ScatBassEngine::pushInput (const juce::AudioBuffer<float>& input)
{
    if (! inputFifo)
        return;
    const int samples = input.getNumSamples();
    if (static_cast<int> (inputScratch.size()) < samples)
        return;
    std::fill_n (inputScratch.begin(), samples, 0.0f);
    const int channels = std::max (1, input.getNumChannels());
    for (int channel = 0; channel < input.getNumChannels(); ++channel)
    {
        const float* source = input.getReadPointer (channel);
        for (int sample = 0; sample < samples; ++sample)
            inputScratch[static_cast<std::size_t> (sample)] +=
                source[sample] / static_cast<float> (channels);
    }
    inputFifo->write (inputScratch.data(), samples);
    workAvailable.signal();
}

void ScatBassEngine::pullOutput (juce::AudioBuffer<float>& output)
{
    const int samples = output.getNumSamples();
    if (static_cast<int> (inputScratch.size()) < samples || ! outputFifo)
    {
        output.clear();
        return;
    }
    const int read = outputFifo->read (inputScratch.data(), samples);
    std::fill (inputScratch.begin() + read,
               inputScratch.begin() + samples,
               0.0f);
    for (int channel = 0; channel < output.getNumChannels(); ++channel)
        std::copy_n (
            inputScratch.data(), samples, output.getWritePointer (channel));
}

void ScatBassEngine::run()
{
    while (! threadShouldExit())
    {
        bool rendered = false;
        while (renderOneFrame())
            rendered = true;
        if (! rendered)
            workAvailable.wait (4);
    }
}

bool ScatBassEngine::processAubio (const float* hop)
{
    std::copy_n (hop, aubioHopSize, aubioInput->data);
    aubio_onset_do (onsetDetector, aubioInput, aubioOutput);
    return aubioOutput->data[0] > 0.0f;
}

int ScatBassEngine::resolveArticulation (float bassF0) const
{
    switch (style.load (std::memory_order_acquire))
    {
        case 0: return 0; // Finger
        case 1: return 1; // Muted
        case 2: return 2; // Pick
        case 3: return bassF0 >= 82.40689f ? 3 : 4; // Slap Auto
        case 4: return 3; // Pop
        case 5: return 4; // Thumb
        case 6: return 5; // Dead note
        default: return 0;
    }
}

int ScatBassEngine::onsetModelArticulation (int bassArticulation) const
{
    // The onset model includes FS_BE at index 5; FS_DN is index 6.
    return bassArticulation == 5 ? 6 : bassArticulation;
}

void ScatBassEngine::updateNoteState (bool onsetCandidate,
                                      bool noiseGate,
                                      float periodicity)
{
    currentOnset = false;
    currentOffset = false;
    const bool retriggerAllowed = framesSinceOnset >= 5; // 80 ms
    if (noteActive)
    {
        if (onsetCandidate && retriggerAllowed)
        {
            currentOffset = true;
            currentOnset = true;
        }
        else if (! noiseGate
                 || (noteAgeSeconds >= 0.080f && periodicity < 0.35f))
        {
            currentOffset = true;
            noteActive = false;
        }
    }
    else if (onsetCandidate && noiseGate)
    {
        currentOnset = true;
    }

    if (currentOnset)
    {
        noteActive = true;
        noteAgeSeconds = 0.0f;
        framesSinceOnset = 0;
        onsetEnvelopeFrame = 0;
        activeArticulation = resolveArticulation (currentF0);
        const float f0Control = std::log2 (std::max (currentF0, 40.0f) / 40.0f)
                                / std::log2 (600.0f / 40.0f);
        onsetEnvelope = models.inferOnsetEnvelope (
            onsetModelArticulation (activeArticulation),
            f0Control,
            currentLoudness);
    }
    currentGate = noteActive;
}

bool ScatBassEngine::renderOneFrame()
{
    if (! inputFifo || ! outputFifo
        || ! inputFifo->peekAndAdvance (
            hostFrame.data(), frameSizeAtHostRate, hopSizeAtHostRate))
        return false;

    inputResampler.process (hostRate / modelSampleRate,
                            hostFrame.data(),
                            modelFrame.data(),
                            modelFrameSize);
    const float* latestHop = modelFrame.data() + modelFrameSize - modelHopSize;
    const float frameRms = rms (latestHop, modelHopSize);
    const float rmsDb = 20.0f * std::log10 (std::max (frameRms, 1.0e-7f));

    if (calibrationFrames < 32)
    {
        noisePeakDb = std::max (noisePeakDb, rmsDb);
        ++calibrationFrames;
    }
    const bool rawNoiseGate =
        calibrationFrames >= 32 && rmsDb > noisePeakDb + 10.0f;
    if (rawNoiseGate)
        gateReleaseFrames = 5;
    else if (gateReleaseFrames > 0)
        --gateReleaseFrames;
    const bool noiseGate = rawNoiseGate || gateReleaseFrames > 0;

    const bool firstSubHopOnset = processAubio (latestHop);
    const bool secondSubHopOnset = processAubio (latestHop + aubioHopSize);
    const bool onsetCandidate =
        (firstSubHopOnset || secondSubHopOnset) && noiseGate;

    const auto started = std::chrono::steady_clock::now();
    const auto crepe = models.inferCrepe (modelFrame);
    currentPeriodicity = clamp (crepe.periodicity, 0.0f, 1.0f);
    if (currentPeriodicity >= 0.10f)
        currentF0 = clamp (crepe.f0Hz * 0.5f, 30.0f, 330.0f);
    else if (! noteActive)
        currentF0 = 0.0f;

    const float standardDeviation = std::sqrt (
        std::max (loudnessVariance, 1.0f));
    currentLoudness = clamp (
        (rmsDb - loudnessMean) / standardDeviation, -3.0f, 3.0f);
    if (noiseGate)
    {
        constexpr float alpha = 0.02f;
        const float difference = rmsDb - loudnessMean;
        loudnessMean += alpha * difference;
        loudnessVariance =
            (1.0f - alpha) * (loudnessVariance + alpha * difference * difference);
    }

    updateNoteState (onsetCandidate, noiseGate, currentPeriodicity);
    const float onsetStrength =
        onsetEnvelopeFrame < static_cast<int> (onsetEnvelope.size())
            ? onsetEnvelope[static_cast<std::size_t> (onsetEnvelopeFrame)]
            : 0.0f;
    const auto controller = models.inferController (
        currentF0,
        currentLoudness,
        activeArticulation,
        onsetStrength,
        currentOffset ? 1.0f : 0.0f,
        currentGate ? 1.0f : 0.0f,
        noteAgeSeconds,
        currentPeriodicity);
    synthesize (
        controller, currentF0, currentLoudness, currentPeriodicity);
    const auto elapsed = std::chrono::duration<float, std::milli> (
        std::chrono::steady_clock::now() - started)
                             .count();

    outputResampler.process (modelSampleRate / hostRate,
                             synthesisFrame.data(),
                             hostOutputFrame.data(),
                             hopSizeAtHostRate);
    outputFifo->write (hostOutputFrame.data(), hopSizeAtHostRate);

    if (noteActive)
    {
        noteAgeSeconds += modelHopSize / static_cast<float> (modelSampleRate);
        ++onsetEnvelopeFrame;
    }
    ++framesSinceOnset;
    publishTelemetry (frameRms, elapsed);
    return true;
}

void ScatBassEngine::buildNoiseImpulse (const std::array<float, 65>& bands)
{
    std::array<float, 128> impulse {};
    for (int sample = 0; sample < 128; ++sample)
    {
        float value = bands[0]
                      + bands[64] * std::cos (pi * sample);
        for (int bin = 1; bin < 64; ++bin)
            value += 2.0f * bands[static_cast<std::size_t> (bin)]
                     * std::cos (2.0f * pi * bin * sample / 128.0f);
        impulse[static_cast<std::size_t> (sample)] = value / 128.0f;
    }
    std::array<float, 256> padded {};
    for (int sample = 0; sample < 128; ++sample)
    {
        const int source = (sample - 64 + 128) % 128;
        const float hann =
            0.5f * (1.0f - std::cos (2.0f * pi * sample / 128.0f));
        padded[static_cast<std::size_t> (sample)] =
            impulse[static_cast<std::size_t> (source)] * hann;
    }
    for (int sample = 0; sample < 256; ++sample)
        noiseKernel[static_cast<std::size_t> (sample)] =
            padded[static_cast<std::size_t> ((sample + 64) % 256)];
}

void ScatBassEngine::synthesize (
    const ModelRuntime::ControllerResult& controls,
    float f0,
    float loudness,
    float periodicity)
{
    buildNoiseImpulse (controls.noiseBands);
    const float loudnessGain =
        clamp (std::pow (10.0f, loudness * 8.0f / 20.0f), 0.15f, 4.0f);
    const float harmonicIndicator =
        1.0f / (1.0f + std::exp (-10.0f * (periodicity - 0.7f)));
    const float harmonicGate = 0.15f + 0.85f * harmonicIndicator;
    const float sustainBranchGain = std::pow (10.0f, 12.0f / 20.0f);
    const float frameOnsetStrength =
        onsetEnvelopeFrame < static_cast<int> (onsetEnvelope.size())
            ? onsetEnvelope[static_cast<std::size_t> (onsetEnvelopeFrame)]
            : 0.0f;

    for (int sample = 0; sample < modelHopSize; ++sample)
    {
        const float position = sample / static_cast<float> (modelHopSize);
        float sustain = 0.0f;
        wavetablePhase += f0 * 512.0f / modelSampleRate;
        while (wavetablePhase >= 512.0f)
            wavetablePhase -= 512.0f;
        const int index0 = static_cast<int> (wavetablePhase);
        const int index1 = (index0 + 1) % 512;
        const float fraction = wavetablePhase - index0;
        for (int table = 0; table < 16; ++table)
        {
            const float weight =
                previousWeights[static_cast<std::size_t> (table)]
                + position
                      * (controls.wavetableWeights[static_cast<std::size_t> (table)]
                         - previousWeights[static_cast<std::size_t> (table)]);
            const float first =
                wavetables[static_cast<std::size_t> (table * 512 + index0)];
            const float second =
                wavetables[static_cast<std::size_t> (table * 512 + index1)];
            sustain += weight * (first + fraction * (second - first));
        }
        const float amplitude =
            previousAmplitude
            + position * (controls.sustainAmplitude - previousAmplitude);
        const float sampleAge =
            noteAgeSeconds + sample / static_cast<float> (modelSampleRate);
        const float fadeIn = clamp (sampleAge / 0.006f, 0.0f, 1.0f);
        sustain *= amplitude * sustainBranchGain * loudnessGain
                   * harmonicGate * fadeIn * (noteActive ? 1.0f : 0.0f);

        noiseHistory[static_cast<std::size_t> (noiseHistoryIndex)] =
            random.nextFloat() * 2.0f - 1.0f;
        float noise = 0.0f;
        for (int tap = 0; tap < 256; ++tap)
        {
            const int history =
                (noiseHistoryIndex - tap + 256) % 256;
            noise += noiseKernel[static_cast<std::size_t> (tap)]
                     * noiseHistory[static_cast<std::size_t> (history)];
        }
        noiseHistoryIndex = (noiseHistoryIndex + 1) % 256;
        noise *= noteActive ? 1.0f : 0.0f;

        float transient = 0.0f;
        const int transientIndex =
            static_cast<int> (sampleAge * modelSampleRate);
        if (noteActive && transientIndex >= 0 && transientIndex < 4800)
        {
            transient =
                transientBank[static_cast<std::size_t> (
                    activeArticulation * 4800 + transientIndex)];
            float window = 1.0f;
            if (sampleAge > 0.26f)
            {
                const float fade =
                    clamp ((0.30f - sampleAge) / 0.04f, 0.0f, 1.0f);
                window = fade * fade * (3.0f - 2.0f * fade);
            }
            const float velocity = 0.25f + 0.75f * frameOnsetStrength;
            transient *= controls.transientGain * window * velocity;
        }
        synthesisFrame[static_cast<std::size_t> (sample)] =
            sustain + noise + transient;
    }
    previousWeights = controls.wavetableWeights;
    previousAmplitude = controls.sustainAmplitude;
}

void ScatBassEngine::publishTelemetry (float inputLevel, float inferenceMs)
{
    telemetryInput.store (inputLevel, std::memory_order_relaxed);
    telemetryF0.store (currentF0, std::memory_order_relaxed);
    telemetryPeriodicity.store (
        currentPeriodicity, std::memory_order_relaxed);
    telemetryOnset.store (currentOnset ? 1.0f : 0.0f, std::memory_order_relaxed);
    telemetryOffset.store (
        currentOffset ? 1.0f : 0.0f, std::memory_order_relaxed);
    telemetryGate.store (currentGate ? 1.0f : 0.0f, std::memory_order_relaxed);
    telemetryAge.store (noteAgeSeconds, std::memory_order_relaxed);
    telemetryArticulation.store (
        activeArticulation, std::memory_order_relaxed);
    telemetryInferenceMs.store (inferenceMs, std::memory_order_relaxed);
}

ScatBassEngine::Telemetry ScatBassEngine::getTelemetry() const
{
    return {
        telemetryInput.load (std::memory_order_relaxed),
        telemetryF0.load (std::memory_order_relaxed),
        telemetryPeriodicity.load (std::memory_order_relaxed),
        telemetryOnset.load (std::memory_order_relaxed),
        telemetryOffset.load (std::memory_order_relaxed),
        telemetryGate.load (std::memory_order_relaxed),
        telemetryAge.load (std::memory_order_relaxed),
        telemetryArticulation.load (std::memory_order_relaxed),
        telemetryInferenceMs.load (std::memory_order_relaxed)
    };
}
