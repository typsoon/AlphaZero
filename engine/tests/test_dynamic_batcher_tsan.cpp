// ThreadSanitizer regression test for the DynamicBatcher buffer-slot race
// (engine/inference/basic_infer.cpp, DynamicBatcher::worker_loop()).
//
// The bug: worker_loop() used to call prepare_batch(tasks, slot) - which
// WRITES directly into buffer_slots[slot]'s pinned memory - before waiting
// for slot_busy[slot] to clear. If `worker` produced batches faster than
// `gpu_executor_thread` could drain them, it could lap all kNumBufferSlots
// (3) slots and start overwriting a slot's pinned buffer while
// gpu_executor_thread still had an in-flight read of that same memory for a
// previous batch - a genuine data race, observed in production as a glibc
// `_int_malloc` heap-corruption abort. The fix splits the wait: worker now
// blocks on `!slot_busy[slot]` BEFORE calling prepare_batch(), not just
// before the ready_batch handoff.
//
// This test relies on ThreadSanitizer to catch the race deterministically,
// rather than on the rare/nondeterministic heap-corruption crash. See
// engine/CMakeLists.txt's ALPHAZERO_BUILD_TSAN_TEST option for how this
// binary is built (its own target, compiling basic_infer.cpp,
// inference_cache.cpp and connect4.cpp directly as its own sources, all with
// -fsanitize=thread - NOT linked against the normal `game`/`inference`
// static libraries, which aren't instrumented).
//
// Design notes:
//
//  - Goes through the *public* interface only (NetworkInfererFactory ->
//    NetworkInferer::infer()), exactly as real self-play code does (see
//    training/self_play.cpp) - DynamicBatcher's class body is private to
//    basic_infer.cpp's translation unit and isn't reachable from here (no
//    friend declaration needed; the race is fully exercisable without one).
//
//  - Runs entirely on CPU (torch::kCPU), not CUDA. The race is fundamentally
//    a CPU-thread synchronization bug over a shared host buffer, not
//    anything GPU-specific - and CUDA would actually make it a *worse* fit
//    for ThreadSanitizer: on the CUDA path, buffer_slots[slot].pinned_buffer
//    is read via an async cudaMemcpyAsync issued from gpu_executor_thread,
//    but the actual memory read happens on the CUDA driver's DMA engine, not
//    via instrumented CPU instructions - TSan cannot see it, so a real race
//    there could easily go unreported. On CPU, Tensor::to(same_device) is a
//    no-op (aliases the same storage, no copy), so the batch tensor that
//    gpu_executor_thread's forward pass reads from is literally the same
//    memory prepare_batch() writes into - no async indirection to hide the
//    race behind.
//
//  - The "network" plugged into DynamicBatcher is a tiny custom op
//    (tsan_test_ops::read_and_zero below), not a real trained model, for two
//    reasons:
//     1. If the read of the input tensor happened inside a prebuilt
//        ATen/libtorch kernel (e.g. `x.sum()`), and that prebuilt library
//        wasn't itself compiled with -fsanitize=thread (confirmed true for
//        the pip-distributed libtorch build used here), TSan would not
//        instrument that read, and the race would go undetected despite
//        being real. Using our own op means the read happens in a plain
//        C++ loop in *this* file, compiled with -fsanitize=thread, which
//        TSan is guaranteed to see.
//     2. It lets one specific batch's execution be held open for a long,
//        controlled time (see the slow-batch mechanism below) without
//        affecting any other batch's latency.
//
//    The op's "forward" method is wired up by hand-building its TorchScript
//    Graph and inserting calls to the op via Graph::insert() (see
//    build_dummy_network()), rather than via
//    torch::jit::script::Module::define()'s TorchScript-source parser: that
//    parser's native (non-Python) resolver in this libtorch build cannot
//    resolve `torch.ops.<ns>.<op>` attribute chains at all (verified
//    directly - it fails identically for builtin ops like torch.ops.aten.add,
//    not just custom ones), even though the op is correctly registered with
//    the dispatcher. Graph::insert() looks the op up directly via the
//    dispatcher/schema registry and sidesteps that resolver entirely.
//
//  - Reproducing the lapping condition deterministically (rather than
//    hoping for it statistically) took a bit of care. The obvious approach
//    - many threads hammering infer() concurrently - does NOT reliably lap
//    the 3 slots: every infer() call blocks its calling thread until its
//    own batch's promise resolves, so with N producer threads all
//    submitting as fast as they can, worker_loop tends to fold most/all of
//    them into a single batch (it drains everything currently pending on
//    each pass), gpu_executor_thread fulfills that whole batch's promises,
//    and all N threads wake and resubmit together - a self-synchronizing
//    "thundering herd" that settles into exactly one batch in flight at a
//    time, however many threads or how much random per-submission jitter is
//    used (verified empirically up to 160 threads). There's simply never a
//    backlog to lap with, because arrival rate is coupled to completion
//    rate in this closed-loop pattern.
//
//    Instead, this test constructs the lapping condition explicitly and
//    deterministically:
//      1. A "slow" request is submitted alone, so it forms its own
//         single-item batch in isolation, occupying whatever slot
//         worker_loop's round-robin is currently on for a long, fixed
//         duration (kSlowDelayUs) - it's tagged via a one-shot atomic flag
//         (tsan_test_ops::slow_calls_remaining) that the custom op consumes,
//         rather than anything game-state-specific.
//      2. After a short head start (long enough for that submission to be
//         drained into its own batch and start executing, but far shorter
//         than kSlowDelayUs), three more "fast" requests are submitted in
//         sequence, each with a small stagger between them so worker_loop
//         has time to drain and hand off each one as its own separate
//         batch. Slots are handed out round-robin, so these three land on
//         the other two slots and then - critically - back on the slow
//         request's slot, while it's still executing.
//    This is repeated for kTrials independent trials to also give the test
//    reasonable stress-test coverage, not just a single roll of the dice.

#include "basic_inferer.hpp"
#include "game.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <connect4.hpp>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <spdlog/spdlog.h>
#include <thread>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

namespace tsan_test_ops {

// One-shot marker consumed by the next read_and_zero() call: sets that one
// batch's delay to kSlowDelayUs (baked in via a constant at graph-build
// time is the same for every call, so instead the *duration actually slept*
// is chosen per-call based on this flag) instead of the normal fast delay.
// A simple atomic rather than anything tied to game state/content, since
// which physical batch is "the slow one" is decided by the test driver's
// submission order, not by what's in it.
std::atomic<int> slow_calls_remaining{0};

// Genuinely reads every element of x - see file header for why this must be
// our own instrumented code rather than an ATen op. Returns zeros shaped
// like a policy row; the actual values are irrelevant, only the read matters.
torch::Tensor read_and_zero(const torch::Tensor &x, int64_t action_dim, int64_t fast_delay_us,
                            int64_t slow_delay_us) {
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "expected float32 input");
    auto contig = x.contiguous();
    const float *data = contig.data_ptr<float>();
    volatile float sink = 0.0f; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
    for (int64_t i = 0; i < contig.numel(); ++i) {
        sink += data[i];
    }
    (void)sink;

    int64_t delay_us = fast_delay_us;
    int expected = slow_calls_remaining.load();
    while (expected > 0) {
        if (slow_calls_remaining.compare_exchange_weak(expected, expected - 1)) {
            delay_us = slow_delay_us;
            break;
        }
    }
    spdlog::info("TSAN_DEBUG: read_and_zero batch_size={} delay_us={}", x.size(0), delay_us);
    if (delay_us > 0) {
        // Keeps gpu_executor_thread inside execute_tensor_batch (hence this
        // batch's slot marked busy) for a long time, giving the "fast"
        // requests below a wide, deterministic window to lap back around to
        // the same slot - see file header.
        std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
    }
    return torch::zeros({x.size(0), action_dim});
}

torch::Tensor make_value(const torch::Tensor &x) {
    return torch::zeros({x.size(0)});
}

} // namespace tsan_test_ops

TORCH_LIBRARY(tsan_test_ops, m) {
    m.def("read_and_zero(Tensor x, int action_dim, int fast_delay_us, int slow_delay_us) -> "
          "Tensor",
          tsan_test_ops::read_and_zero);
    m.def("make_value(Tensor x) -> Tensor", tsan_test_ops::make_value);
}

namespace {

// DynamicBatcher only requires a torch::jit::script::Module with a compiled
// `forward` method (see basic_infer.cpp:
// infer_method(network->get_method("forward"))). Built by hand-assembling
// its Graph and inserting calls to the tsan_test_ops:: custom ops above
// (rather than via Module::define()'s TorchScript-source parser - see file
// header for why that path doesn't work here) so the "forward pass" is
// really just our own instrumented C++ code.
torch::jit::script::Module build_dummy_network(int64_t action_dim, int64_t fast_delay_us,
                                               int64_t slow_delay_us) {
    torch::jit::script::Module module("TsanDummyNet");

    auto graph = std::make_shared<torch::jit::Graph>();
    auto *self_val = graph->addInput("self");
    self_val->setType(module.type());
    auto *x_val = graph->addInput("x");
    x_val->setType(torch::TensorType::get());

    auto *action_dim_const = graph->insertConstant(action_dim);
    auto *fast_delay_const = graph->insertConstant(fast_delay_us);
    auto *slow_delay_const = graph->insertConstant(slow_delay_us);
    auto *policy = graph->insert(c10::Symbol::fromQualString("tsan_test_ops::read_and_zero"),
                                 {x_val, action_dim_const, fast_delay_const, slow_delay_const});
    auto *value = graph->insert(c10::Symbol::fromQualString("tsan_test_ops::make_value"), {x_val});

    auto *tuple_out = graph->insertNode(graph->createTuple({policy, value}))->output();
    graph->registerOutput(tuple_out);

    auto *fn = module._ivalue()->compilation_unit()->create_function("forward", graph);
    module.type()->addMethod(fn);
    return module;
}

void run_one_submission(NetworkInfererFactory &factory, std::atomic<bool> &failed) {
    auto inferer = factory.get_inferer();
    Connect4 game;
    std::vector<const GameState *> states{&game};
    try {
        auto results = inferer->infer(states);
        if (results.size() != 1 || results[0].legal_actions.size() != 7) {
            spdlog::error("unexpected inference result shape");
            failed = true;
        }
    } catch (const std::exception &e) {
        spdlog::error("infer() threw: {}", e.what());
        failed = true;
    }
}

} // namespace

int main() {
    constexpr int64_t kActionDim = Connect4::action_dim; // 7
    constexpr int64_t kFastDelayUs = 0;
    // Long enough to comfortably dominate the few hundred us to low
    // milliseconds it takes worker_loop to drain and hand off 3 more
    // single-item batches, even accounting for ThreadSanitizer's overhead on
    // every mutex/condition_variable operation involved.
    constexpr int64_t kSlowDelayUs = 50000;
    constexpr int kHeadStartUs = 3000;
    constexpr int kStaggerUs = 1500;
    constexpr int kWaitForCount = 1; // fire batches as soon as any task exists
    constexpr int kTimeoutMs = 1;
    constexpr int kTrials = 40;
    // One persistent thread per role (slow, fast1, fast2, fast3), signaled
    // via atomics rather than spawned fresh per trial: creating a brand new
    // std::thread under ThreadSanitizer (which has to set up its own
    // per-thread shadow state on top of the real clone()/pthread_create())
    // measured as slow and, worse, *highly variable* - multiple milliseconds
    // some of the time - which was swamping the deliberately tight
    // kHeadStartUs/kStaggerUs staggering below and collapsing the intended
    // 4-separate-batches pattern back into the same "everything bundles
    // into one or two batches" problem this whole scheme exists to avoid.
    // Long-lived threads that just spin on an atomic "go" signal react in
    // (low-single-digit) microseconds instead.
    constexpr int kRoles = 4; // 0 = slow, 1..3 = fast

    auto module = build_dummy_network(kActionDim, kFastDelayUs, kSlowDelayUs);
    auto model_path = (std::filesystem::temp_directory_path() / "az_tsan_dummy_net.pt").string();
    module.save(model_path);

    NetworkInfererFactory factory(model_path, torch::kCPU, kWaitForCount, kTimeoutMs,
                                  /*transposition_cache_entries=*/0);

    std::atomic<bool> failed{false};
    std::array<std::atomic<int>, kRoles> go_signal{};
    std::array<std::atomic<int>, kRoles> done_count{};
    for (int r = 0; r < kRoles; ++r) {
        go_signal[r].store(0);
        done_count[r].store(0);
    }

    std::vector<std::thread> role_threads;
    role_threads.reserve(kRoles);
    for (int r = 0; r < kRoles; ++r) {
        role_threads.emplace_back([&, r]() {
            for (int trial = 1; trial <= kTrials; ++trial) {
                while (go_signal[r].load(std::memory_order_acquire) != trial) {
                    std::this_thread::yield();
                }
                run_one_submission(factory, failed);
                done_count[r].fetch_add(1, std::memory_order_release);
            }
        });
    }

    for (int trial = 1; trial <= kTrials; ++trial) {
        tsan_test_ops::slow_calls_remaining.store(1);
        go_signal[0].store(trial, std::memory_order_release); // fire the slow request
        std::this_thread::sleep_for(std::chrono::microseconds(kHeadStartUs));
        for (int r = 1; r < kRoles; ++r) { // targets slot+1, slot+2, slot+0 (reused)
            go_signal[r].store(trial, std::memory_order_release);
            std::this_thread::sleep_for(std::chrono::microseconds(kStaggerUs));
        }
        for (int r = 0; r < kRoles; ++r) {
            while (done_count[r].load(std::memory_order_acquire) != trial) {
                std::this_thread::yield();
            }
        }
    }
    for (auto &th : role_threads) {
        th.join();
    }

    if (failed.load()) {
        spdlog::error("DynamicBatcher TSan stress test failed");
        return 1;
    }

    spdlog::info("DynamicBatcher TSan stress test: {} trials completed with consistent results. "
                 "Absence/presence of a data race is reported by ThreadSanitizer itself (via "
                 "stderr and the process exit code), not by this program's own checks.",
                 kTrials);
    return 0;
}
