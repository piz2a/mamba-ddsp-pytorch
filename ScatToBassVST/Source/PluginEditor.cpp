#include "PluginEditor.h"

#include <ScatWebData.h>

#include <unordered_map>

namespace
{
std::vector<std::byte> streamBytes (juce::InputStream& stream)
{
    std::vector<std::byte> result (
        static_cast<std::size_t> (stream.getTotalLength()));
    stream.setPosition (0);
    stream.read (result.data(), result.size());
    return result;
}

const char* mimeType (const juce::String& extension)
{
    static const std::unordered_map<juce::String, const char*> types {
        { "html", "text/html" },
        { "js", "text/javascript" },
        { "css", "text/css" },
        { "svg", "image/svg+xml" },
        { "png", "image/png" },
        { "woff2", "font/woff2" },
        { "json", "application/json" }
    };
    if (const auto found = types.find (extension.toLowerCase());
        found != types.end())
        return found->second;
    return "application/octet-stream";
}

std::vector<std::byte> webFile (const juce::String& request)
{
    juce::MemoryInputStream zipStream {
        scat_web::scat_to_bass_web_zip,
        scat_web::scat_to_bass_web_zipSize,
        false
    };
    juce::ZipFile zip { zipStream };
    static const juce::String prefix = [] {
        juce::MemoryInputStream input {
            scat_web::scat_to_bass_web_zip,
            scat_web::scat_to_bass_web_zipSize,
            false
        };
        juce::ZipFile archive { input };
        for (int index = 0; index < archive.getNumEntries(); ++index)
            if (const auto* entry = archive.getEntry (index);
                entry != nullptr && entry->filename.endsWith ("index.html"))
                return entry->filename.dropLastCharacters (10);
        return juce::String {};
    }();
    if (const auto* entry = zip.getEntry (prefix + request))
        if (auto stream = zip.createStreamForEntry (*entry))
            return streamBytes (*stream);
    return {};
}
}

ScatToBassAudioProcessorEditor::ScatToBassAudioProcessorEditor (
    ScatToBassAudioProcessor& owner)
    : AudioProcessorEditor (&owner),
      processor (owner),
      styleAttachment (
          *processor.parameters().getParameter ("style"), styleRelay, nullptr),
      browser (
          juce::WebBrowserComponent::Options {}
              .withBackend (
                  juce::WebBrowserComponent::Options::Backend::defaultBackend)
              .withWinWebView2Options (
                  juce::WebBrowserComponent::Options::WinWebView2 {}
                      .withUserDataFolder (
                          juce::File::getSpecialLocation (
                              juce::File::tempDirectory)))
              .withResourceProvider (
                  [this] (const auto& url) { return getResource (url); })
              .withOptionsFrom (styleRelay)
              .withNativeIntegrationEnabled())
{
    addAndMakeVisible (browser);
    setResizable (false, false);
    setSize (720, 500);
    browser.goToURL (
        juce::WebBrowserComponent::getResourceProviderRoot());
    startTimerHz (30);
}

void ScatToBassAudioProcessorEditor::resized()
{
    browser.setBounds (getLocalBounds());
}

void ScatToBassAudioProcessorEditor::timerCallback()
{
    const auto frame = processor.telemetry();
    auto object = juce::DynamicObject::Ptr { new juce::DynamicObject };
    object->setProperty ("inputLevel", frame.inputLevel);
    object->setProperty ("f0Hz", frame.f0Hz);
    object->setProperty ("periodicity", frame.periodicity);
    object->setProperty ("onset", frame.onset);
    object->setProperty ("offset", frame.offset);
    object->setProperty ("gate", frame.gate);
    object->setProperty ("noteAge", frame.noteAge);
    object->setProperty ("articulation", frame.articulation);
    object->setProperty ("inferenceMs", frame.inferenceMs);
    browser.emitEventIfBrowserIsVisible ("controlFrame", object.get());
}

std::optional<juce::WebBrowserComponent::Resource>
ScatToBassAudioProcessorEditor::getResource (const juce::String& url) const
{
    const auto request =
        url == "/" ? juce::String { "index.html" }
                   : url.fromFirstOccurrenceOf ("/", false, false);
    auto bytes = webFile (request);
    if (bytes.empty())
        return std::nullopt;
    return juce::WebBrowserComponent::Resource {
        std::move (bytes),
        mimeType (request.fromLastOccurrenceOf (".", false, false))
    };
}
