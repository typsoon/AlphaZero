// AddressSanitizer/UBSan stress harness for the ChessEncoderV2History
// self-play path (engine/game/chess_encoder_v2history.cpp + the Chess
// position-history hook in engine/game/chess.cpp + the DynamicBatcher
// inference pipeline in engine/inference/basic_infer.cpp).
//
// Motivation: during bring-up of ChessEncoderV2History a self-play smoke test
// once printed `malloc(): unsorted double linked list corrupted` on process
// exit (10 clean / 1 corrupted, never reproduced on demand). That signature is
// heap-metadata corruption - an out-of-bounds write or a double-free - which
// AddressSanitizer catches deterministically at the instruction that does it,
// unlike the rare non-deterministic abort. This binary drives the exact code
// paths that the failing smoke test exercised:
//
//   - Chess::move_piece()'s history_boards / history_repetitions_before /
//     history_count push (advancing each game by real random legal moves so
//     those arrays actually fill and the encoder reads them),
//   - ChessEncoderV2History::write_canonical_state() into BOTH the batcher's
//     pinned buffer AND (with the transposition cache enabled below) the
//     per-thread `scratch` std::vector in NetworkInferer::infer() - the latter
//     is a plain heap allocation ASan fully instruments with redzones, so an
//     off-by-one in the 63-plane write is caught there directly,
//   - the multi-threaded DynamicBatcher slot pipeline under a genuine backlog
//     (many producer threads), plus repeated factory construct/destruct cycles
//     to exercise the worker/gpu-thread startup+teardown path that the
//     process-exit corruption pointed at.
//
// Runs on CPU with a hand-built dummy TorchScript net (same technique and
// rationale as tests/test_dynamic_batcher_tsan.cpp - see that file's header):
// the network body is irrelevant to a memory-safety hunt, only that the input
// tensor of the encoder-produced size is written by our instrumented encoder
// and read back. Built as its own ASan target from an explicit source list
// (ASan is incompatible with the main build's flags), see engine/CMakeLists.txt
// ALPHAZERO_BUILD_ASAN_STRESS_TEST.
//
// Exit code: 0 if all iterations completed with consistent result shapes and
// ASan/UBSan reported nothing (ASan aborts the process with a non-zero code the
// moment it sees corruption). Args: [outer_cycles] [threads] [submits_per_thread].

#include "basic_inferer.hpp"
#include "chess.hpp"
#include "chess_encoder_v2history.hpp"
#include "game.hpp"
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <random>
#include <spdlog/spdlog.h>
#include <thread>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

namespace asan_stress_ops {

// Reads every element of x in plain instrumented C++ (so an OOB read of the
// batch buffer is caught here rather than hidden inside an uninstrumented ATen
// kernel - see test_dynamic_batcher_tsan.cpp), then returns a zero policy row.
torch::Tensor policy_head(const torch::Tensor &x, int64_t action_dim) {
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "expected float32 input");
    auto contig = x.contiguous();
    const float *data = contig.data_ptr<float>();
    volatile float sink = 0.0f; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
    for (int64_t i = 0; i < contig.numel(); ++i) {
        sink += data[i];
    }
    (void)sink;
    return torch::zeros({x.size(0), action_dim});
}

torch::Tensor value_head(const torch::Tensor &x) {
    return torch::zeros({x.size(0)});
}

} // namespace asan_stress_ops

TORCH_LIBRARY(asan_stress_ops, m) {
    m.def("policy_head(Tensor x, int action_dim) -> Tensor", asan_stress_ops::policy_head);
    m.def("value_head(Tensor x) -> Tensor", asan_stress_ops::value_head);
}

namespace {

torch::jit::script::Module build_dummy_network(int64_t action_dim) {
    torch::jit::script::Module module("AsanStressDummyNet");
    auto graph = std::make_shared<torch::jit::Graph>();
    auto *self_val = graph->addInput("self");
    self_val->setType(module.type());
    auto *x_val = graph->addInput("x");
    x_val->setType(torch::TensorType::get());

    auto *action_dim_const = graph->insertConstant(action_dim);
    auto *policy = graph->insert(c10::Symbol::fromQualString("asan_stress_ops::policy_head"),
                                 {x_val, action_dim_const});
    auto *value =
        graph->insert(c10::Symbol::fromQualString("asan_stress_ops::value_head"), {x_val});
    auto *tuple_out = graph->insertNode(graph->createTuple({policy, value}))->output();
    graph->registerOutput(tuple_out);

    auto *fn = module._ivalue()->compilation_unit()->create_function("forward", graph);
    module.type()->addMethod(fn);
    return module;
}

// Advances `game` by up to `max_moves` random legal moves (stopping early at a
// terminal position), so its history_boards window fills before it is encoded.
void play_random_moves(Chess &game, std::mt19937 &rng, int max_moves) {
    for (int m = 0; m < max_moves; ++m) {
        auto legal = game.get_legal_actions();
        if (legal.empty())
            break;
        std::uniform_int_distribution<size_t> pick(0, legal.size() - 1);
        game.step(legal[pick(rng)]);
    }
}

void producer(NetworkInfererFactory &factory, int submits, unsigned seed,
              std::atomic<bool> &failed) {
    std::mt19937 rng(seed);
    auto inferer = factory.get_inferer();
    for (int s = 0; s < submits && !failed.load(); ++s) {
        // A fresh game advanced to a random depth (0..40 plies), plus a couple
        // of clones taken at successive depths so one submitted batch holds
        // several DISTINCT history-window contents at once - the mix the real
        // MCTS leaf batches feed the encoder.
        std::uniform_int_distribution<int> depth_dist(0, 40);
        auto base = std::make_shared<Chess>();
        play_random_moves(*base, rng, depth_dist(rng));

        std::vector<std::shared_ptr<Chess>> owned;
        owned.push_back(base);
        int extra = std::uniform_int_distribution<int>(0, 3)(rng);
        for (int e = 0; e < extra; ++e) {
            auto clone = std::static_pointer_cast<Chess>(base->clone());
            play_random_moves(*clone, rng, std::uniform_int_distribution<int>(0, 6)(rng));
            owned.push_back(clone);
        }

        std::vector<const GameState *> states;
        states.reserve(owned.size());
        for (const auto &g : owned)
            states.push_back(g.get());

        try {
            auto results = inferer->infer(states);
            if (results.size() != states.size()) {
                spdlog::error("result count {} != states {}", results.size(), states.size());
                failed = true;
                return;
            }
            for (size_t i = 0; i < states.size(); ++i) {
                if (results[i].legal_actions != states[i]->get_legal_actions()) {
                    spdlog::error("legal-action mismatch at index {}", i);
                    failed = true;
                    return;
                }
            }
        } catch (const std::exception &e) {
            spdlog::error("infer() threw: {}", e.what());
            failed = true;
            return;
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    int outer_cycles = (argc > 1) ? std::atoi(argv[1]) : 8;
    int threads = (argc > 2) ? std::atoi(argv[2]) : 8;
    int submits = (argc > 3) ? std::atoi(argv[3]) : 40;

    constexpr int64_t kActionDim = Chess::action_dim;
    constexpr int kHistory = 4;

    auto module = build_dummy_network(kActionDim);
    auto model_path = (std::filesystem::temp_directory_path() / "az_asan_stress_net.pt").string();
    module.save(model_path);

    std::atomic<bool> failed{false};

    for (int cycle = 0; cycle < outer_cycles && !failed.load(); ++cycle) {
        // A fresh factory each cycle: exercises worker/gpu-thread startup and -
        // critically for the process-exit corruption signature - teardown (the
        // factory destructor joins those threads and releases the pinned
        // buffers + transposition cache) repeatedly, not just once.
        auto encoder = std::make_shared<ChessEncoderV2History>(kHistory);
        NetworkInfererFactory factory(model_path, torch::kCPU, /*wait_for_count=*/2,
                                      /*timeout_ms=*/1,
                                      /*transposition_cache_entries=*/4096, encoder);

        std::vector<std::thread> pool;
        pool.reserve(threads);
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back(producer, std::ref(factory), submits,
                              static_cast<unsigned>(cycle * 1000 + t + 1), std::ref(failed));
        }
        for (auto &th : pool)
            th.join();
        spdlog::info("ASan stress: cycle {}/{} done", cycle + 1, outer_cycles);
    }

    if (failed.load()) {
        spdlog::error("ASan self-play stress test FAILED (logic error; ASan reports memory "
                      "errors separately via process abort)");
        return 1;
    }
    spdlog::info("ASan self-play stress test: {} cycles x {} threads x {} submits completed "
                 "cleanly. Any heap corruption would have aborted via ASan.",
                 outer_cycles, threads, submits);
    return 0;
}
