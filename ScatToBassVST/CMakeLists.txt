cmake_minimum_required(VERSION 3.22)
project(ScatToBass VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_definitions(JUCE_VST3_CAN_REPLACE_VST2=0)

set(JUCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../JUCE" CACHE PATH "JUCE source tree")
set(AUBIO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../aubio" CACHE PATH "aubio source tree")
if(APPLE)
    set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime macOS arm64/x86_64 package")
else()
    set(
        ONNXRUNTIME_ROOT
        "${CMAKE_CURRENT_SOURCE_DIR}/../ddsp_onnx_benchmark/vendor/onnxruntime-linux-x64-1.18.1"
        CACHE PATH
        "ONNX Runtime package"
    )
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "Set ONNXRUNTIME_ROOT to an ONNX Runtime C/C++ package")
endif()
find_library(
    ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)
add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED GLOBAL)
set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ui/dist/index.html")
    message(FATAL_ERROR "Build the React UI first: npm --prefix ui run build")
endif()

add_subdirectory("${JUCE_ROOT}" JUCE)
add_subdirectory("${AUBIO_ROOT}" aubio-build EXCLUDE_FROM_ALL)

set(PLUGIN_FORMATS VST3 Standalone)
if(APPLE)
    list(APPEND PLUGIN_FORMATS AU)
endif()

juce_add_plugin(ScatToBass
    COMPANY_NAME "Jihoaudio"
    BUNDLE_ID "com.jihoaudio.scattobass"
    PLUGIN_MANUFACTURER_CODE Jiho
    PLUGIN_CODE Stbs
    FORMATS ${PLUGIN_FORMATS}
    PRODUCT_NAME "Scat to Bass"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE
    COPY_PLUGIN_AFTER_BUILD FALSE
    VST3_CAN_REPLACE_VST2 FALSE
)

target_sources(ScatToBass PRIVATE
    Source/ModelRuntime.cpp
    Source/ModelRuntime.h
    Source/ScatBassEngine.cpp
    Source/ScatBassEngine.h
    Source/PluginProcessor.cpp
    Source/PluginProcessor.h
    Source/PluginEditor.cpp
    Source/PluginEditor.h
)

set(WEBVIEW_ZIP "${CMAKE_CURRENT_BINARY_DIR}/scat_to_bass_web.zip")
file(
    ARCHIVE_CREATE
    OUTPUT "${WEBVIEW_ZIP}"
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/ui/dist"
    FORMAT zip
)
juce_add_binary_data(
    ScatToBassWeb
    NAMESPACE scat_web
    HEADER_NAME ScatWebData.h
    SOURCES "${WEBVIEW_ZIP}"
)
juce_add_binary_data(ScatToBassModels
    NAMESPACE scat_models
    HEADER_NAME ScatModelData.h
    SOURCES
        Models/bass_ddsp_controller.onnx
        Models/torchcrepe_tiny.onnx
        Models/onset_envelope.onnx
        Models/sustain_wavetables.f32
        Models/transient_bank.f32
        Models/manifest.json
)

add_executable(ScatToBassModelSmoke
    tools/model_smoke.cpp
    Source/ModelRuntime.cpp
    Source/ModelRuntime.h
)
target_include_directories(ScatToBassModelSmoke PRIVATE Source)
target_link_libraries(ScatToBassModelSmoke PRIVATE
    ScatToBassModels
    ONNXRuntime::ONNXRuntime
)
set_target_properties(ScatToBassModelSmoke PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)

add_executable(ScatToBassEngineSmoke
    tools/engine_smoke.cpp
    Source/ModelRuntime.cpp
    Source/ModelRuntime.h
    Source/ScatBassEngine.cpp
    Source/ScatBassEngine.h
)
target_include_directories(ScatToBassEngineSmoke PRIVATE
    Source
    "${AUBIO_ROOT}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/aubio-build/src"
)
target_link_libraries(ScatToBassEngineSmoke PRIVATE
    ScatToBassModels
    ONNXRuntime::ONNXRuntime
    aubio
    juce::juce_audio_basics
    juce::juce_core
    juce::juce_recommended_config_flags
)
target_compile_definitions(ScatToBassEngineSmoke PRIVATE JUCE_USE_CURL=0)
set_target_properties(ScatToBassEngineSmoke PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${CMAKE_CURRENT_BINARY_DIR}/aubio-build/src"
)

target_include_directories(ScatToBass PRIVATE
    Source
    "${AUBIO_ROOT}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/aubio-build/src"
)
target_link_libraries(ScatToBass PRIVATE
    ScatToBassWeb
    ScatToBassModels
    aubio
    ONNXRuntime::ONNXRuntime
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_dsp
    juce::juce_gui_extra
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags
)

target_compile_definitions(ScatToBass PRIVATE
    JUCE_WEB_BROWSER=1
    JUCE_USE_CURL=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
)

if(UNIX AND NOT APPLE)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GTK3 REQUIRED IMPORTED_TARGET gtk+-x11-3.0)
    pkg_check_modules(WEBKIT2 REQUIRED IMPORTED_TARGET webkit2gtk-4.0)
    target_link_libraries(ScatToBass PRIVATE PkgConfig::GTK3 PkgConfig::WEBKIT2)
    set_target_properties(ScatToBass PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
    )
endif()
if(APPLE)
    set_target_properties(ScatToBass PROPERTIES
        BUILD_RPATH "@loader_path/../../../../Frameworks;${ONNXRUNTIME_ROOT}/lib"
        INSTALL_RPATH "@loader_path/../../../../Frameworks"
    )
endif()
