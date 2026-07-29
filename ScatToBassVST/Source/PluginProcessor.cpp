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
