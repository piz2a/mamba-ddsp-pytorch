#pragma once

#include "ScatBassEngine.h"

#include <juce_audio_processors/juce_audio_processors.h>

class ScatToBassAudioProcessor : public juce::AudioProcessor
{
public:
    ScatToBassAudioProcessor();
    ~ScatToBassAudioProcessor() override = default;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.3; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}
    void getStateInformation (juce::MemoryBlock&) override;
    void setStateInformation (const void*, int) override;

    juce::AudioProcessorValueTreeState& parameters() { return state; }
    ScatBassEngine::Telemetry telemetry() const { return engine.getTelemetry(); }

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout makeParameters();

    ScatBassEngine engine;
    juce::AudioProcessorValueTreeState state;
    std::atomic<float>* styleParameter = nullptr;
    std::atomic<float>* noiseGateThresholdParameter = nullptr;
    std::atomic<float>* octaveShiftParameter = nullptr;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ScatToBassAudioProcessor)
};
