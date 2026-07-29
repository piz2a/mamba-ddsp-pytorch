#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace {

using Clock = std::chrono::steady_clock;

struct Summary {
    double mean;
    double stddev;
    double minimum;
    double p50;
    double p95;
    double p99;
    double maximum;
};

double percentile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    const double position = q * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

Summary summarize(const std::vector<double>& values) {
    const double mean =
        std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double variance = 0.0;
    for (const double value : values) {
        variance += (value - mean) * (value - mean);
    }
    variance /= values.size();
    return {
        mean,
        std::sqrt(variance),
        *std::min_element(values.begin(), values.end()),
        percentile(values, 0.50),
        percentile(values, 0.95),
        percentile(values, 0.99),
        *std::max_element(values.begin(), values.end()),
    };
}

void benchmarkModel(
    Ort::Env& env,
    const std::string& modelName,
    const std::string& path,
    const std::string& provider,
    int warmup,
    int iterations,
    std::ofstream& samples,
    std::ofstream& summaryFile) {
    std::cout << "Loading " << modelName << " from " << path << std::endl;
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (provider != "cpu") {
        throw std::runtime_error(
            "This portable build currently benchmarks CPUExecutionProvider only.");
    }
    Ort::Session session(env, path.c_str(), options);
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const bool bass = modelName == "Bass-DDSP";
    const bool dwts = modelName == "Vanilla DWTS";
    float pitch[] = {55.0f};
    float loudness[] = {0.0f};
    int64_t articulation[] = {0};
    float onsetStrength[] = {1.0f};
    float offset[] = {0.0f};
    float gate[] = {1.0f};
    float noteAge[] = {0.0f};
    float periodicity[] = {0.85f};
    std::vector<float> hiddenState(256, 0.0f);
    const int64_t scalarShape[] = {1, 1, 1};
    const int64_t articulationShape[] = {1, 1};
    const int64_t hiddenShape[] = {1, 1, 256};

    std::vector<std::string> inputNameStorage = {"pitch", "loudness"};
    if (bass) {
        inputNameStorage.insert(
            inputNameStorage.end(),
            {"articulation", "onset_strength", "offset", "gate", "note_age",
             "periodicity"});
    }
    inputNameStorage.push_back("hidden_state");
    std::vector<const char*> inputNames;
    std::vector<Ort::Value> inputValues;
    inputValues.emplace_back(Ort::Value::CreateTensor<float>(
        memory, pitch, 1, scalarShape, 3));
    inputValues.emplace_back(Ort::Value::CreateTensor<float>(
        memory, loudness, 1, scalarShape, 3));
    if (bass) {
        inputValues.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory, articulation, 1, articulationShape, 2));
        inputValues.emplace_back(Ort::Value::CreateTensor<float>(
            memory, onsetStrength, 1, scalarShape, 3));
        inputValues.emplace_back(Ort::Value::CreateTensor<float>(
            memory, offset, 1, scalarShape, 3));
        inputValues.emplace_back(Ort::Value::CreateTensor<float>(
            memory, gate, 1, scalarShape, 3));
        inputValues.emplace_back(Ort::Value::CreateTensor<float>(
            memory, noteAge, 1, scalarShape, 3));
        inputValues.emplace_back(Ort::Value::CreateTensor<float>(
            memory, periodicity, 1, scalarShape, 3));
    }
    inputValues.emplace_back(Ort::Value::CreateTensor<float>(
        memory, hiddenState.data(), hiddenState.size(), hiddenShape, 3));
    for (const auto& name : inputNameStorage) {
        inputNames.push_back(name.c_str());
    }

    std::vector<std::string> outputNameStorage;
    if (bass) {
        outputNameStorage = {
            "wavetable_weights", "sustain_amplitude", "noise_bands",
            "transient_gain", "next_hidden_state"};
    } else if (dwts) {
        outputNameStorage = {
            "wavetable_weights", "sustain_amplitude", "noise_bands",
            "next_hidden_state"};
    } else {
        outputNameStorage = {
            "harmonic_parameters", "noise_bands", "next_hidden_state"};
    }
    std::vector<const char*> outputNames;
    for (const auto& name : outputNameStorage) {
        outputNames.push_back(name.c_str());
    }

    auto run = [&]() {
        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            inputNames.data(),
            inputValues.data(),
            inputValues.size(),
            outputNames.data(),
            outputNames.size());
        const float* nextHidden = outputs.back().GetTensorData<float>();
        std::copy(nextHidden, nextHidden + hiddenState.size(), hiddenState.begin());
        return outputs;
    };
    for (int index = 0; index < warmup; ++index) {
        auto outputs = run();
    }

    std::vector<double> timings;
    timings.reserve(iterations);
    for (int index = 0; index < iterations; ++index) {
        const auto start = Clock::now();
        auto outputs = run();
        const auto end = Clock::now();
        const double milliseconds =
            std::chrono::duration<double, std::milli>(end - start).count();
        timings.push_back(milliseconds);
        samples << modelName << ',' << provider << ',' << index << ','
                << std::setprecision(12) << milliseconds << '\n';
    }
    const Summary result = summarize(timings);
    summaryFile << modelName << ',' << provider << ',' << iterations << ','
                << result.mean << ',' << result.stddev << ',' << result.minimum << ','
                << result.p50 << ',' << result.p95 << ',' << result.p99 << ','
                << result.maximum << ',' << result.p50 / 16.0 << ','
                << result.p99 / 16.0 << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ddsp_onnx_benchmark MODELS_DIR OUTPUT_DIR\n";
        return 2;
    }
    const std::string models = argv[1];
    const std::string output = argv[2];
    std::ofstream samples(output + "/onnx_cpp_latency_samples.csv");
    std::ofstream summary(output + "/onnx_cpp_summary.csv");
    samples << "model,provider,iteration,latency_ms\n";
    summary << "model,provider,iterations,mean_ms,std_ms,min_ms,p50_ms,p95_ms,"
               "p99_ms,max_ms,rtf_p50,rtf_p99\n";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ddsp-controller-benchmark");
    const std::vector<std::pair<std::string, std::string>> modelPaths = {
        {"Bass-DDSP", models + "/bass_ddsp_controller.onnx"},
        {"Vanilla DWTS", models + "/vanilla_dwts_controller.onnx"},
        {"Vanilla DDSP", models + "/vanilla_ddsp_controller.onnx"},
    };
    try {
        for (const auto& [name, path] : modelPaths) {
            benchmarkModel(env, name, path, "cpu", 500, 5000, samples, summary);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
