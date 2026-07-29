#include "PluginProcessor.h"
#include "PluginEditor.h"

ScatToBassAudioProcessor::ScatToBassAudioProcessor()
    : AudioProcessor (
          BusesProperties()
              .withInput ("Vocal Input", juce::AudioChannelSet::stereo(), true)
              .withOutput ("Bass Output", juce::AudioChannelSet::stereo(), true)),
      state (*this, nullptr, "ScatToBassState", makeParameters())
{
    styleParameter = state.getRawParameterValue ("style");
    noiseGateThresholdParameter =
        state.getRawParameterValue ("noiseGateThreshold");
    octaveShiftParameter = state.getRawParameterValue ("octaveShift");
}

juce::AudioProcessorValueTreeState::ParameterLayout
ScatToBassAudioProcessor::makeParameters()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;
    layout.add (std::make_unique<juce::AudioParameterChoice> (
        juce::ParameterID { "style", 1 },
        "Playing Style",
        juce::StringArray {
            "Finger",
            "Muted",
            "Pick",
            "Slap Auto",
            "Slap Pop",
            "Slap Thumb",
            "Dead Note"
        },
        0));
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { "noiseGateThreshold", 1 },
        "Noise Gate Threshold",
        juce::NormalisableRange<float> { -80.0f, 0.0f, 0.5f },
        -45.0f,
        juce::AudioParameterFloatAttributes {}
            .withLabel ("dB")
            .withStringFromValueFunction ([] (float value, int) {
                return juce::String (value, 1);
            })));
    layout.add (std::make_unique<juce::AudioParameterInt> (
        juce::ParameterID { "octaveShift", 1 },
        "Octave Shift",
        -2,
        2,
        0,
        juce::AudioParameterIntAttributes {}
            .withLabel ("oct")
            .withStringFromValueFunction ([] (int value, int) {
                return value > 0 ? "+" + juce::String (value)
                                 : juce::String (value);
            })));
    return layout;
}

void ScatToBassAudioProcessor::prepareToPlay (double sampleRate,
                                               int samplesPerBlock)
{
    engine.prepare (sampleRate, samplesPerBlock);
    setLatencySamples (engine.latencySamples());
}

void ScatToBassAudioProcessor::releaseResources()
{
    engine.reset();
}

bool ScatToBassAudioProcessor::isBusesLayoutSupported (
    const BusesLayout& layouts) const
{
    const auto output = layouts.getMainOutputChannelSet();
    return (output == juce::AudioChannelSet::mono()
            || output == juce::AudioChannelSet::stereo())
           && layouts.getMainInputChannelSet() == output;
}

void ScatToBassAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                              juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    engine.setStyle (juce::roundToInt (styleParameter->load()));
    engine.setNoiseGateThresholdDb (noiseGateThresholdParameter->load());
    engine.setOctaveShift (juce::roundToInt (octaveShiftParameter->load()));
    engine.pushInput (buffer);
    engine.pullOutput (buffer);
}

void ScatToBassAudioProcessor::getStateInformation (
    juce::MemoryBlock& destination)
{
    const auto xml = state.copyState().createXml();
    copyXmlToBinary (*xml, destination);
}

void ScatToBassAudioProcessor::setStateInformation (const void* data,
                                                     int bytes)
{
    if (const auto xml = getXmlFromBinary (data, bytes))
        if (xml->hasTagName (state.state.getType()))
            state.replaceState (juce::ValueTree::fromXml (*xml));
}

juce::AudioProcessorEditor* ScatToBassAudioProcessor::createEditor()
{
    return new ScatToBassAudioProcessorEditor (*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new ScatToBassAudioProcessor();
}
