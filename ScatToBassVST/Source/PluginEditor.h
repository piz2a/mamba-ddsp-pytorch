#pragma once

#include "PluginProcessor.h"

#include <juce_gui_extra/juce_gui_extra.h>

class ScatToBassAudioProcessorEditor : public juce::AudioProcessorEditor,
                                       private juce::Timer
{
public:
    explicit ScatToBassAudioProcessorEditor (ScatToBassAudioProcessor&);
    ~ScatToBassAudioProcessorEditor() override = default;

    void resized() override;
    void paint (juce::Graphics&) override {}

private:
    void timerCallback() override;
    std::optional<juce::WebBrowserComponent::Resource> getResource (
        const juce::String&) const;

    ScatToBassAudioProcessor& processor;
    juce::WebSliderRelay styleRelay { "style" };
    juce::WebSliderParameterAttachment styleAttachment;
    juce::WebBrowserComponent browser;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (
        ScatToBassAudioProcessorEditor)
};
