// Minimal repro: does the "Memcpy DtoH (Device -> Pageable)" stall under
// CUDAGraphs mode persist when calling the TRT-compiled model directly from C++
// (torch::jit::load + Method::operator()), bypassing DynamicBatcher/MCTS/self_play
// entirely? Mirrors temporary_utils/test_cudagraphs_pinned_input.py exactly (same
// pinned-buffer input pattern, same batch-size sequence) to isolate whether the
// C++ invocation path itself is the remaining variable.
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <atomic>
#include <cmath>
#include <iostream>
#include <optional>
#include <random>
#include <thread>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/script.h>
#include <torch/torch.h>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <network_path> <kineto_out>\n";
        return 1;
    }
    std::string network_path = argv[1];
    std::string kineto_out = argv[2];

    torch::Device device(torch::kCUDA);
    auto network = std::make_shared<torch::jit::script::Module>(
        torch::jit::load(network_path, device));
    network->to(device);
    network->eval();
    auto infer_method = network->get_method("forward");

    auto op = c10::Dispatcher::singleton().findSchema({"tensorrt::set_cudagraphs_mode", ""});
    if (!op.has_value()) {
        std::cerr << "tensorrt::set_cudagraphs_mode op not found\n";
        return 1;
    }
    op->typed<void(int64_t)>().call(int64_t(1)); // SUBGRAPH_CUDAGRAPHS
    std::cout << "CUDAGraphs mode enabled\n";

    auto pinned = torch::empty({64, 19, 8, 8},
                               torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));

    bool with_stream_guard = (argc >= 4);
    bool from_worker_thread = (argc >= 5);

    auto run_calls = [&]() {
        torch::NoGradGuard no_grad;
        auto x = pinned.slice(0, 0, 32).to(device, /*non_blocking=*/true);
        infer_method({x});
        torch::cuda::synchronize();

        torch::profiler::impl::ProfilerConfig config(torch::profiler::impl::ProfilerState::KINETO,
                                                     false, false, false, false, false);
        std::set<torch::profiler::impl::ActivityType> activities = {
            torch::profiler::impl::ActivityType::CPU, torch::profiler::impl::ActivityType::CUDA};
        torch::autograd::profiler::prepareProfiler(config, activities);
        torch::autograd::profiler::enableProfiler(config, activities);

        std::optional<c10::cuda::CUDAStream> copy_stream;
        torch::Tensor pinned_index;
        torch::Tensor pinned_gathered;
        std::mt19937 gen(0);
        std::uniform_int_distribution<int> dist(1, 64);
        std::uniform_int_distribution<int> action_dist(0, 20479);
        for (int i = 0; i < 300; ++i) {
            int batch_size = dist(gen);
            auto x_i = pinned.slice(0, 0, batch_size).to(device, /*non_blocking=*/true);
            auto result = infer_method({x_i});

            if (with_stream_guard) {
                // Mirrors execute_tensor_batch's post-inference stream handoff exactly:
                // record an event on whatever stream was current during infer_method,
                // move to a dedicated copy_stream, and make it wait on that event.
                at::cuda::CUDAEvent infer_done;
                infer_done.record();
                if (!copy_stream.has_value()) {
                    copy_stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false,
                                                               device.index());
                }
                c10::cuda::CUDAStreamGuard stream_guard(*copy_stream);
                infer_done.block(*copy_stream);

                auto outputs = result.toTuple()->elements();
                auto policy_gpu = outputs[0].toTensor();
                auto value_gpu = outputs[1].toTensor();

                // Mirrors execute_tensor_batch's actual sparse-gather path exactly:
                // build a padded int64 index tensor (~35 "legal actions" per row,
                // like chess), H2D it, gather on-device, D2H the gathered result -
                // instead of just a plain .to(kCPU) on the full output.
                constexpr int64_t kMaxActions = 40;
                if (!pinned_index.defined() || pinned_index.size(0) < batch_size) {
                    pinned_index = torch::zeros(
                        {64, kMaxActions},
                        torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
                    pinned_gathered = torch::empty(
                        {64, kMaxActions},
                        torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
                }
                auto index_host = pinned_index.slice(0, 0, batch_size);
                auto *index_ptr = index_host.data_ptr<int64_t>();
                for (int64_t r = 0; r < batch_size; ++r)
                    for (int64_t c = 0; c < kMaxActions; ++c)
                        index_ptr[r * kMaxActions + c] = action_dist(gen);

                auto index_gpu = index_host.to(device, /*non_blocking=*/true);
                auto gathered_gpu = policy_gpu.gather(1, index_gpu);
                auto gathered_dst = pinned_gathered.slice(0, 0, batch_size);
                gathered_dst.copy_(gathered_gpu, /*non_blocking=*/true);
                auto value_host = value_gpu.to(torch::kCPU, /*non_blocking=*/true);
                copy_stream->synchronize();
            }
        }
        torch::cuda::synchronize();

        auto profiler_result = torch::autograd::profiler::disableProfiler();
        profiler_result->save(kineto_out);
        std::cout << "Kineto profile saved to " << kineto_out << "\n";
    };

    bool with_busy_cpu_thread = (argc >= 6);
    std::atomic<bool> stop_busy_thread{false};
    std::thread busy_thread;
    if (with_busy_cpu_thread) {
        // Mirrors the real workload's remaining concurrency: even with
        // thread_count=1, a separate CPU-bound self-play/MCTS thread (tree search,
        // move generation) always runs concurrently with DynamicBatcher's worker
        // thread doing the GPU calls. Busy-spin doing real floating point work
        // (not just an empty loop, which a compiler/scheduler might treat very
        // differently) to approximate that.
        busy_thread = std::thread([&stop_busy_thread]() {
            volatile double x = 0.0;
            while (!stop_busy_thread.load(std::memory_order_relaxed)) {
                for (int i = 0; i < 100000; ++i)
                    x += std::sqrt(static_cast<double>(i) + 1.0);
            }
        });
    }

    if (from_worker_thread) {
        // Mirrors DynamicBatcher's architecture exactly: infer_method is always
        // called from a dedicated spawned std::thread, never from main()'s own
        // thread.
        std::thread worker(run_calls);
        worker.join();
    } else {
        run_calls();
    }

    if (with_busy_cpu_thread) {
        stop_busy_thread.store(true, std::memory_order_relaxed);
        busy_thread.join();
    }
    return 0;
}
