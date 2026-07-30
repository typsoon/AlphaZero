// Tests for the InferenceBackend dispatch added to engine/inference/basic_infer.cpp:
// NetworkInfererFactory loads either a TorchScript module (plain scripted or
// torch_tensorrt-compiled - both ZIP archives) via TorchScriptInferenceBackend,
// or, when this binary was built with TensorRT SDK headers
// (ALPHAZERO_NATIVE_TRT_ENGINE - see engine/inference/CMakeLists.txt), a raw
// TensorRT engine (network.py's backend="onnx" output) via
// TensorRTInferenceBackend.
//
// DynamicBatcher/get_network_func/the backend classes are all private to
// basic_infer.cpp's translation unit (see test_dynamic_batcher_tsan.cpp's
// design notes for why - same reasoning applies here), so these tests go
// through the public interface only (NetworkInfererFactory ->
// NetworkInferer::infer()), exactly as real self-play code does.
//
// Fixtures (a tiny scripted TorchScript module, and - when compiled in - a
// tiny native TensorRT engine) are built programmatically at test time rather
// than depending on a real trained checkpoint being present in the repo,
// matching this codebase's existing test philosophy (see
// test_dynamic_batcher_tsan.cpp's "not a real trained model" design note).
// Both fixtures are wired to Connect4Encoder's exact input shape (1x6x7) and
// a real Connect4 GameState (the actual initial position, 7 legal actions),
// so the full production pipeline - encoding, batching, inference, gather -
// is exercised end to end, not just the backend classes in isolation.

#include "connect4.hpp"
#include "connect4_encoder.hpp"
#include "inference/basic_inferer.hpp"
#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakWarningPlugin.h>
#include <CppUTest/TestHarness.h>
#include <cstdio>
#include <fstream>
#include <memory>
#include <torch/script.h>
#include <torch/torch.h>
#include <unistd.h>

#ifdef ALPHAZERO_NATIVE_TRT_ENGINE
#include <NvInfer.h>
#endif

namespace {

std::string temp_path(const std::string &suffix) {
    return "/tmp/test_inference_backend_" + std::to_string(getpid()) + suffix;
}

// A tiny TorchScript module with the real network's (policy, value) forward()
// contract: policy is the input flattened to [B, 42] (well above Connect4's
// action_dim=7, so gather() over legal actions 0-6 is always in range), value
// is the sum of every element per row, reshaped to [B, 1]. Plain aten ops
// (reshape/sum), not a custom op, so TorchScript's native .define() resolver
// handles it directly - no hand-built Graph needed (contrast
// test_dynamic_batcher_tsan.cpp, which specifically needs a custom op and so
// can't use .define()).
std::string build_torchscript_fixture() {
    torch::jit::script::Module module("m");
    module.define(R"JIT(
def forward(self, x):
    b = x.shape[0]
    policy = x.reshape(b, -1)
    value = x.sum(dim=[1, 2, 3], keepdim=True).reshape(b, 1)
    return policy, value
)JIT");
    auto path = temp_path(".pt_scripted");
    module.save(path);
    return path;
}

#ifdef ALPHAZERO_NATIVE_TRT_ENGINE
class FixtureLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kERROR) {
            fprintf(stderr, "[TRT fixture] %s\n", msg);
        }
    }
};

// A native TensorRT engine with the same (policy, value) contract as
// build_torchscript_fixture() above: policy = input flattened to [B, 42],
// value = sum of every element per row (kept as [B,1,1,1] - contiguous, so
// it reads identically to [B,1] as a flat float array, matching how
// execute_tensor_batch consumes it). Built entirely via TensorRT's C++
// builder API (no ONNX involved), so this test has no Python dependency.
std::string build_native_trt_fixture() {
    static FixtureLogger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));

    nvinfer1::Dims input_dims{};
    input_dims.nbDims = 4;
    input_dims.d[0] = -1; // dynamic batch
    input_dims.d[1] = 1;
    input_dims.d[2] = 6;
    input_dims.d[3] = 7;
    auto *input = network->addInput("input", nvinfer1::DataType::kFLOAT, input_dims);

    auto *shuffle = network->addShuffle(*input);
    nvinfer1::Dims policy_dims{};
    policy_dims.nbDims = 2;
    policy_dims.d[0] = -1;
    policy_dims.d[1] = 42;
    shuffle->setReshapeDimensions(policy_dims);
    shuffle->getOutput(0)->setName("policy");
    network->markOutput(*shuffle->getOutput(0));

    // Reduce every axis except batch (bit 0): bits 1,2,3 set = 0b1110 = 14.
    auto *reduce = network->addReduce(*input, nvinfer1::ReduceOperation::kSUM,
                                       /*reduceAxes=*/14U, /*keepDimensions=*/true);
    reduce->getOutput(0)->setName("value");
    network->markOutput(*reduce->getOutput(0));

    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    auto *profile = builder->createOptimizationProfile();
    for (auto selector : {nvinfer1::OptProfileSelector::kMIN, nvinfer1::OptProfileSelector::kOPT,
                          nvinfer1::OptProfileSelector::kMAX}) {
        nvinfer1::Dims d = input_dims;
        d.d[0] = (selector == nvinfer1::OptProfileSelector::kMAX) ? 4 : 1;
        profile->setDimensions("input", selector, d);
    }
    config->addOptimizationProfile(profile);

    std::unique_ptr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
        throw std::runtime_error("test fixture: failed to build native TRT engine");
    }

    auto path = temp_path(".pt_trt");
    std::ofstream f(path, std::ios::binary);
    f.write(static_cast<const char *>(serialized->data()),
            static_cast<std::streamsize>(serialized->size()));
    return path;
}
#endif

// Runs one real inference through `network_path` (Connect4, the actual
// initial position) and returns the single result. encoder is passed
// explicitly so both fixtures - which are wired to Connect4Encoder's exact
// [1,6,7] shape - are used regardless of default-encoder auto-detection.
std::vector<inference_result> infer_initial_position(const std::string &network_path,
                                                      torch::Device device) {
    NetworkInfererFactory factory(network_path, device, /*wait_for_count=*/1,
                                  /*timeout_ms=*/10, /*transposition_cache_entries=*/0,
                                  std::make_shared<Connect4Encoder>());
    auto inferer = factory.get_inferer();
    Connect4 state;
    std::vector<const GameState *> states{&state};
    return inferer->infer(states);
}

} // namespace

TEST_GROUP(InferenceBackendTests){
    // Every test here loads a real torch::jit::Module or TensorRT engine,
    // which allocates through libtorch's/TensorRT's own long-lived internal
    // registries (op schemas, JIT compiler caches, plan caches) - correctly
    // process-lifetime singletons, not bugs, but CppUTest's leak detector
    // (running with thread-safe tracking on - see main() below, needed to
    // avoid a real crash from CUDA/TensorRT's internal worker threads racing
    // an unprotected tracking list) reports them as leaks per test regardless.
    void setup() override {
        MemoryLeakWarningPlugin::getFirstPlugin()->ignoreAllLeaksInTest();
    }
};

TEST(InferenceBackendTests, TorchScriptBackendInfersCorrectly) {
    auto path = build_torchscript_fixture();
    auto results = infer_initial_position(path, torch::kCPU);
    std::remove(path.c_str());

    CHECK_EQUAL(1, static_cast<int>(results.size()));
    const auto &r = results[0];
    // Connect4's initial position has all 7 columns open.
    CHECK_EQUAL(7, static_cast<int>(r.legal_actions.size()));
    for (int a = 0; a < 7; ++a) {
        CHECK_EQUAL(a, r.legal_actions[a]);
    }
    // The fixture's policy = input.reshape(B,-1), so column c's logit is
    // simply the input tensor's c-th element in row-major order - for
    // Connect4's initial (empty) board every cell is 0.
    for (float logit : r.legal_action_logits) {
        DOUBLES_EQUAL(0.0, logit, 1e-6);
    }
    // value = sum of all 42 input elements - also 0 on an empty board.
    DOUBLES_EQUAL(0.0, r.value, 1e-6);
}

TEST(InferenceBackendTests, NonZipFileFailsWithATensorRTRelatedError) {
    // Neither a ZIP/TorchScript archive nor (when ALPHAZERO_NATIVE_TRT_ENGINE
    // is off) a loadable format at all - garbage bytes. Exercises
    // get_network_func's ZIP-magic dispatch: without native TRT support this
    // must fail with the specific "raw TensorRT engine, not a loadable
    // TorchScript archive" diagnostic (not a generic/cryptic error); with it,
    // TensorRTInferenceBackend's deserializeCudaEngine legitimately fails on
    // non-engine bytes and throws its own TensorRT-specific message. Both
    // mention "TensorRT", so one assertion covers both configurations. Uses
    // torch::kCUDA (not kCPU): TensorRTInferenceBackend's constructor takes a
    // c10::cuda::CUDAGuard before it even opens the file, which throws its
    // own (TensorRT-unrelated) error on a non-CUDA device - realistic anyway,
    // since TensorRT is CUDA-only and checkpoint_manager.py only ever
    // attempts this when torch.cuda.is_available().
    auto path = temp_path(".pt_trt_garbage");
    std::ofstream f(path, std::ios::binary);
    f << "not a zip archive and not a real tensorrt engine either";
    f.close();

    bool threw = false;
    try {
        NetworkInfererFactory factory(path, torch::kCUDA);
        (void)factory;
    } catch (const std::exception &e) {
        threw = true;
        std::string what = e.what();
        CHECK_TRUE(what.find("TensorRT") != std::string::npos);
    }
    std::remove(path.c_str());
    CHECK_TRUE(threw);
}

#ifdef ALPHAZERO_NATIVE_TRT_ENGINE
TEST(InferenceBackendTests, NativeTensorRTBackendInfersCorrectly) {
    auto path = build_native_trt_fixture();
    auto results = infer_initial_position(path, torch::kCUDA);
    std::remove(path.c_str());

    CHECK_EQUAL(1, static_cast<int>(results.size()));
    const auto &r = results[0];
    CHECK_EQUAL(7, static_cast<int>(r.legal_actions.size()));
    for (int a = 0; a < 7; ++a) {
        CHECK_EQUAL(a, r.legal_actions[a]);
    }
    // Same fixture semantics as the TorchScript test above: an empty initial
    // board encodes to all zeros, so every policy logit and the value are 0.
    for (float logit : r.legal_action_logits) {
        DOUBLES_EQUAL(0.0, logit, 1e-3); // fp16-execution-scale tolerance
    }
    DOUBLES_EQUAL(0.0, r.value, 1e-3);
}
#endif

int main(int argc, char **argv) {
    // Real torch::jit::load/TensorRT calls allocate through libtorch's own
    // long-lived internal registries (op schemas, JIT compiler caches) -
    // CppUTest's new/delete interception flags those as "leaks" even though
    // they're correctly process-lifetime singletons, not bugs. Same fix as
    // test_inference_cache.cpp.
    //
    // CppUTest's leak-detector list is NOT thread-safe by default (turning
    // that on is this separate opt-in call) - CUDA/TensorRT spin up their own
    // internal worker threads, whose allocations through CppUTest's
    // globally-overridden new/delete would then race unprotected against the
    // main thread's on the same linked list. That matches exactly what was
    // observed: NativeTensorRTBackendInfersCorrectly's test body (which does
    // real TensorRT engine build+execution) always ran fine, but
    // postTestAction's leak-count walk right after it consistently segfaulted
    // (gdb: MemoryLeakDetectorList::isInPeriod on a bad node) - i.e. list
    // corruption during the test, not a bug in the test logic itself, and
    // neither turnOffNewDeleteOverloads() nor ignoreAllLeaksInTest() (tried
    // first) touch this since the corruption already happened by the time
    // either would apply.
    MemoryLeakWarningPlugin::turnOnThreadSafeNewDeleteOverloads();
    // CommandLineTestRunner::RunAllTests (not the RUN_ALL_TESTS macro,
    // despite test_dual_inferer.cpp using it - that file doesn't touch real
    // torch/CUDA allocations, so it never hit this): the macro's path still
    // ran MemoryLeakWarningPlugin::postTestAction post-test, which segfaulted
    // walking the leak-detector's node list after this test's heavy
    // TensorRT/CUDA allocation, even with the overloads turned off above.
    // Calling RunAllTests directly is the pattern test_inference_cache.cpp
    // already uses successfully alongside turnOffNewDeleteOverloads().
    return CommandLineTestRunner::RunAllTests(argc, argv);
}
