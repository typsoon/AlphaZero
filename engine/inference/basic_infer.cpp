#include "basic_inferer.hpp"
#include "inferer.hpp"
#include <ATen/core/TensorBody.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <condition_variable>
#include <connect4.hpp>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <spdlog/spdlog.h>
#include <thread>
#ifdef ALPHAZERO_TRT_LIB_PATH
#include <dlfcn.h>
#endif

#include <torch/torch.h> // For torch::Tensor and device
#include <utility>
#include <vector>

using Network = torch::jit::script::Module;

class DynamicBatcher {
    std::mutex mtx;
    std::condition_variable state_arrived_cv;
    int wait_for_count;
    int timeout_ms;
    bool stop = false;
    std::thread worker;

    struct Task {
        std::vector<const GameState *> states;
        int count{};
        std::promise<std::vector<inference_result>> promise;
    };

    torch::Tensor pinned_buffer;
    torch::Tensor pinned_index_buffer;
    torch::Tensor pinned_gathered_buffer;
    torch::Tensor pinned_value_buffer;
    std::vector<int64_t> single_shape;

    std::vector<std::shared_ptr<Task>> pending_tasks;
    int current_count = 0;

    std::shared_ptr<Network> network;
    torch::jit::Method infer_method;
    torch::Device device;

    // Dedicated stream for our own post-inference copies/gather, so they don't land
    // on the legacy default stream. Enqueuing work on the legacy default stream
    // implicitly waits for every other stream on the device to drain first (a CUDA
    // backward-compat behavior) - since TensorRT's forward pass runs on its own
    // stream, that meant every one of our copy/gather calls blocked until TensorRT's
    // in-flight kernels finished, even though nothing about the call itself required
    // that (measured: ~3-5ms stalls on ~10-25% of calls, matching TRT's per-batch
    // kernel time). Only used/initialized when device is CUDA.
    std::optional<c10::cuda::CUDAStream> copy_stream;

    std::vector<inference_result>
    execute_tensor_batch(torch::Tensor batched, const std::vector<int> &legal_actions_flat,
                         const std::vector<int> &legal_actions_offsets) {
        if (batched.size(0) == 0)
            return {};
        auto batch_size = batched.size(0);
        batched = batched.to(device);

        std::vector<inference_result> out;
        try {
            torch::NoGradGuard no_grad;

            // Run the forward pass and our own post-processing (gather + D2H copies)
            // on the same dedicated non-default stream throughout, instead of the
            // forward pass on the ambient stream followed by a cross-stream handoff
            // to a separate copy_stream for post-processing. Two distinct problems
            // this avoids:
            //  1. The ambient/default stream is CUDA's legacy default stream, whose
            //     enqueue calls block until *every* other stream on the device
            //     drains (~3-5ms stalls on 10-25% of calls, matching TensorRT's
            //     per-batch kernel time almost exactly - this is what motivated
            //     originally moving post-processing to copy_stream).
            //  2. Consuming policy_gpu/value_gpu from a *different* stream than the
            //     one that produced them - even with correct event-based
            //     synchronization - was itself independently traced (via nsys
            //     backtraces resolving directly to this function) to a ~1.7ms stall
            //     on our own gathered_dst.copy_()/value_dst.copy_() calls, tiny
            //     transfers (~5KB) taking >1000x longer than a properly pinned async
            //     copy of that size should. Running everything on one stream makes
            //     this same-stream by construction - ordering is automatic via
            //     program order, no event handoff needed at all.
            if (!copy_stream.has_value() && device.is_cuda()) {
                copy_stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
            }
            std::optional<c10::cuda::CUDAStreamGuard> stream_guard;
            if (device.is_cuda()) {
                stream_guard.emplace(*copy_stream);
            }

            auto result = infer_method({batched});
            auto outputs = result.toTuple()->elements();
            auto policy_gpu = outputs[0].toTensor();
            auto value_gpu = outputs[1].toTensor();

            // Row lengths vary per state (each has its own legal-action count), so
            // pad to this round's max.
            int64_t max_actions = 0;
            for (int64_t i = 0; i < batch_size; ++i) {
                max_actions = std::max<int64_t>(max_actions, legal_actions_offsets[i + 1] -
                                                                 legal_actions_offsets[i]);
            }

            torch::Tensor value_host;
            torch::Tensor gathered_host;
            // Row stride to use when reading gathered_host below. Usually equals
            // max_actions, except on the CUDA path where the pinned destination
            // buffer's width (which only ever grows) can be wider - see the
            // gather_width assignment below for why using the buffer's actual width
            // instead of max_actions there matters.
            int64_t gather_width = max_actions;
            if (device.is_cuda()) {
                // Transferring the full dense policy row (Chess: 20480 floats) across
                // PCIe for every state when only ~30-40 are ever read is bandwidth-bound
                // (measured ~12.7 GB/s, consistent from 80KB to 18MB transfers - not
                // per-call overhead), so gathering just the needed logits on-device
                // before the D2H copy cuts the payload by roughly the same ~500x that
                // extracting only legal actions already saved on the host-processing
                // side.
                if (!pinned_value_buffer.defined() || pinned_value_buffer.size(0) < batch_size) {
                    auto value_shape = value_gpu.sizes().vec();
                    value_shape[0] = std::max<int64_t>(batch_size, wait_for_count * 2);
                    pinned_value_buffer = torch::empty(
                        value_shape,
                        torch::TensorOptions().dtype(value_gpu.dtype()).pinned_memory(true));
                }
                if (max_actions > 0) {
                    if (!pinned_index_buffer.defined() ||
                        pinned_index_buffer.size(0) < batch_size ||
                        pinned_index_buffer.size(1) < max_actions) {
                        pinned_index_buffer = torch::zeros(
                            {std::max<int64_t>(batch_size, wait_for_count * 2), max_actions},
                            torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
                        pinned_gathered_buffer = torch::empty(
                            {std::max<int64_t>(batch_size, wait_for_count * 2), max_actions},
                            torch::TensorOptions().dtype(policy_gpu.dtype()).pinned_memory(true));
                    }
                    // Use the buffers' actual (already-allocated) width, not this
                    // round's max_actions, for the index/gather/destination shapes
                    // below - the buffers only ever grow (never shrink) to fit the
                    // historical max, so slicing to just max_actions on a
                    // since-grown-wider buffer would leave a stride gap after each
                    // row. Measured: that non-contiguous destination made copy_()
                    // ~1000x slower (a "Memcpy DtoH (Device -> Pageable)" instead of a
                    // fast pinned DMA) despite the underlying storage genuinely being
                    // pinned - traced via nsys backtraces directly to this call.
                    gather_width = pinned_gathered_buffer.size(1);

                    auto index_host = pinned_index_buffer.slice(0, 0, batch_size);
                    auto *index_ptr = index_host.data_ptr<int64_t>();
                    // Padding slots (rows shorter than max_actions, or columns beyond
                    // it up to gather_width) must point at a valid index - the
                    // gathered value there is simply never read below, since we only
                    // ever take the first (offsets[i+1]-offsets[i]) entries per row.
                    std::memset(index_ptr, 0,
                                static_cast<size_t>(batch_size * gather_width) * sizeof(int64_t));
                    for (int64_t i = 0; i < batch_size; ++i) {
                        int begin = legal_actions_offsets[i];
                        int end = legal_actions_offsets[i + 1];
                        for (int j = begin; j < end; ++j) {
                            index_ptr[i * gather_width + (j - begin)] = legal_actions_flat[j];
                        }
                    }

                    // Stream-ordered on the same (default) CUDA stream as the H2D copy
                    // below and the forward pass above, so no manual sync is needed
                    // between them - CUDA guarantees ordering within a stream.
                    auto index_gpu = index_host.to(device, true);
                    auto gathered_gpu = policy_gpu.gather(1, index_gpu);

                    auto gathered_dst = pinned_gathered_buffer.slice(0, 0, batch_size);
                    gathered_dst.copy_(gathered_gpu, true);
                    gathered_host = gathered_dst;
                }

                auto value_dst = pinned_value_buffer.slice(0, 0, batch_size);
                value_dst.copy_(value_gpu, true);
                // Everything above (forward pass, gather, both D2H copies) ran on
                // copy_stream, so waiting on it alone - rather than the whole device -
                // is sufficient and avoids blocking on unrelated device activity.
                copy_stream->synchronize();

                // Below we extract every value any consumer will need into private
                // per-result vectors before this function returns, so gathered_host/
                // value_host never leave this function and never need to be shared
                // across threads - no clone(), pool, or lifetime coordination needed.
                value_host = value_dst.contiguous();
            } else {
                value_host = value_gpu.contiguous();
                if (max_actions > 0) {
                    std::vector<int64_t> index_flat(static_cast<size_t>(batch_size * max_actions),
                                                    0);
                    for (int64_t i = 0; i < batch_size; ++i) {
                        int begin = legal_actions_offsets[i];
                        int end = legal_actions_offsets[i + 1];
                        for (int j = begin; j < end; ++j) {
                            index_flat[i * max_actions + (j - begin)] = legal_actions_flat[j];
                        }
                    }
                    auto index_cpu = torch::from_blob(index_flat.data(), {batch_size, max_actions},
                                                      torch::kInt64);
                    gathered_host = policy_gpu.contiguous().gather(1, index_cpu).contiguous();
                }
            }
            const float *value_ptr = value_host.data_ptr<float>();
            const float *gathered_ptr = max_actions > 0 ? gathered_host.data_ptr<float>() : nullptr;

            out.reserve(batch_size);
            for (int64_t i = 0; i < batch_size; ++i) {
                int begin = legal_actions_offsets[i];
                int end = legal_actions_offsets[i + 1];
                std::vector<int> actions(legal_actions_flat.begin() + begin,
                                         legal_actions_flat.begin() + end);
                std::vector<float> logits(actions.size());
                const float *row = gathered_ptr + i * gather_width;
                for (size_t j = 0; j < actions.size(); ++j) {
                    logits[j] = row[j];
                }
                out.push_back(
                    inference_result{std::move(actions), std::move(logits), value_ptr[i]});
            }
        } catch (const std::exception &e) {
            spdlog::error("Exception caught in execute_tensor_batch: {}", e.what());
            throw;
        }
        return out;
    }

    void process_batch(std::vector<std::shared_ptr<Task>> tasks) {
        int total_count = 0;
        int games_count = 0;
        for (const auto &t : tasks) {
            total_count += t->count;
            games_count += t->count;
        }
        if (total_count == 0)
            return;

        if (single_shape.empty()) {
            for (const auto &t : tasks) {
                if (!t->states.empty()) {
                    single_shape = t->states[0]->get_state_shape();
                    break;
                }
            }
        }

        torch::Tensor games_batch;
        if (games_count > 0) {
            if (!pinned_buffer.defined() || pinned_buffer.size(0) < games_count) {
                auto alloc_shape = single_shape;
                alloc_shape.insert(alloc_shape.begin(), std::max(games_count, wait_for_count * 2));
                auto options = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
                pinned_buffer = torch::empty(alloc_shape, options);
            }

            auto *giant_data = pinned_buffer.data_ptr<float>();
            int state_size = 1;
            for (long i : single_shape)
                state_size *= i;

            int offset = 0;
            for (const auto &t : tasks) {
                for (auto &state : t->states) {
                    state->write_canonical_state(giant_data +
                                                 (static_cast<ptrdiff_t>(offset * state_size)));
                    offset++;
                }
            }
            games_batch = pinned_buffer.slice(0, 0, games_count).to(device, /*non_blocking=*/true);
        }

        torch::Tensor giant_batch = games_batch;

        // Legal actions for every state in the batch, computed here (rather than
        // supplied by the caller) since every GameState can produce its own via
        // get_legal_actions(). Stored contiguously instead of as one std::vector<int>
        // per state: this batch gets rebuilt every MCTS simulation, so keeping it to
        // a handful of allocations regardless of batch size matters.
        std::vector<int> legal_actions_flat;
        std::vector<int> legal_actions_offsets{0};
        legal_actions_offsets.reserve(total_count + 1);
        for (const auto &t : tasks) {
            for (const auto &state : t->states) {
                auto legal_actions = state->get_legal_actions();
                legal_actions_flat.insert(legal_actions_flat.end(), legal_actions.begin(),
                                          legal_actions.end());
                legal_actions_offsets.push_back(static_cast<int>(legal_actions_flat.size()));
            }
        }

        std::vector<inference_result> results;
        try {
            results = execute_tensor_batch(giant_batch, legal_actions_flat, legal_actions_offsets);
        } catch (...) {
            auto ex = std::current_exception();
            for (auto &t : tasks) {
                t->promise.set_exception(ex);
            }
            return;
        }

        int offset = 0;
        for (const auto &t : tasks) {
            std::vector<inference_result> chunk;
            chunk.reserve(t->count);
            for (int i = 0; i < t->count; i++) {
                chunk.push_back(results[offset++]);
            }
            t->promise.set_value(std::move(chunk));
        }
    }

    void worker_loop() {
        while (true) {
            std::vector<std::shared_ptr<Task>> tasks_to_execute;
            {
                std::unique_lock<std::mutex> lock(mtx);
                state_arrived_cv.wait(lock, [this] { return stop || current_count > 0; });

                if (stop && pending_tasks.empty()) {
                    break;
                }

                if (current_count < wait_for_count) {
                    state_arrived_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
                        return stop || current_count >= wait_for_count;
                    });
                }

                if (!pending_tasks.empty()) {
                    tasks_to_execute = std::move(pending_tasks);
                    pending_tasks.clear();
                    current_count = 0;
                }
            }

            if (!tasks_to_execute.empty()) {
                process_batch(std::move(tasks_to_execute));
            }
        }
    }

  public:
    DynamicBatcher(int wait_for_count, int timeout_ms, std::shared_ptr<Network> network,
                   torch::Device device)
        : wait_for_count(wait_for_count), timeout_ms(timeout_ms), network(network),
          infer_method(network->get_method("forward")), device(device) {
        worker = std::thread(&DynamicBatcher::worker_loop, this);
    }

    ~DynamicBatcher() {
        {
            std::scoped_lock lock(mtx);
            stop = true;
        }
        state_arrived_cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }

    std::vector<inference_result> submit(const std::vector<const GameState *> &states) {
        int count = states.size();
        if (count == 0)
            return {};

        auto task = std::make_shared<Task>();
        task->states = states;
        task->count = count;
        auto future = task->promise.get_future();

        {
            std::scoped_lock lock(mtx);
            pending_tasks.push_back(task);
            current_count += count;
        }
        state_arrived_cv.notify_one();
        return future.get();
    }
};

NetworkInferer::NetworkInferer(std::shared_ptr<DynamicBatcher> batcher, torch::Device device)
    : Inferer(device), batcher(std::move(batcher)) {}

vector<inference_result> NetworkInferer::infer(const vector<const GameState *> &states) {
    if (states.empty())
        return {};

    return batcher->submit(states);
}

static std::shared_ptr<Network> get_network_func(std::string network_file_path,
                                                 torch::Device device) {
    if (std::filesystem::exists(network_file_path)) {
        try {
#ifdef ALPHAZERO_TRT_LIB_PATH
            static const bool trt_runtime_loaded = []() {
                void *handle = dlopen(ALPHAZERO_TRT_LIB_PATH, RTLD_NOW | RTLD_GLOBAL);
                if (handle == nullptr) {
                    const char *dlopen_error = dlerror();
                    spdlog::warn("Failed to pre-load libtorchtrt from '{}': {}",
                                 ALPHAZERO_TRT_LIB_PATH,
                                 dlopen_error != nullptr ? dlopen_error : "unknown error");
                    return false;
                }
                return true;
            }();
            (void)trt_runtime_loaded;
#endif
            return std::make_shared<Network>(torch::jit::load(network_file_path, device));
        } catch (const c10::Error &e) {
            spdlog::error(
                "Failed to load network from {}. Ensure it is exported using TorchScript.",
                network_file_path);
            throw std::runtime_error("Network file is not in TorchScript format");
        } catch (const std::exception &e) {
            spdlog::error("Failed to load network: {}", e.what());
            throw std::runtime_error("Failed to load network");
        }
    } else {
        spdlog::error("File {} doesn't exist", network_file_path);
        throw std::runtime_error("Network file not found");
    }
}

#ifdef ALPHAZERO_TRT_LIB_PATH
// Enables TensorRT's CUDAGraphs replay mode for every subsequent execute_engine
// call (a global, non-thread-local flag inside torch_tensorrt's runtime - safe
// here since all inference already funnels through a single DynamicBatcher worker
// thread). The idea: replaying a captured graph instead of re-running execute_engine
// each call should avoid the several-millisecond-per-call "Memcpy DtoH (Device ->
// Pageable)" stall traced (via nsys backtraces) to torch_tensorrt's TorchScript
// execute_engine wrapper.
//
// Measured this DOES NOT help as-is: profiled on the real DynamicBatcher workload
// (12 games/8 threads), the pageable copy persisted at essentially unchanged
// count/cost (~37k calls, ~3.2ms avg) even with CUDAGraphs active (confirmed via
// cudaGraphLaunch/Instantiate counts), and CUDAGraphs added its own overhead
// (graph launch/instantiate/destroy, ~7s total) on top - a net regression, despite
// an isolated single-threaded Python script showing the copy fully eliminated. The
// gap between that isolated test and the real C++ DynamicBatcher path is still
// unexplained (candidates: our get_method("forward") C++ invocation vs Python's
// direct call going through torch_tensorrt's runtime differently, or some
// interaction with the 8 concurrent self-play threads even though only one thread
// ever calls into TensorRT) - not yet root-caused.
//
// Defaults OFF pending that investigation. Set ALPHAZERO_ENABLE_CUDAGRAPHS=1 to
// opt in for testing.
//
// Called via c10's dispatcher (tensorrt::set_cudagraphs_mode, registered by
// libtorchtrt.so's static initializers once it's loaded) rather than linking
// torch_tensorrt's own C++ headers directly - those pull in the full TensorRT SDK
// (NvInfer.h), which isn't installed here; only the pip-distributed runtime
// libraries are.
static void enable_cudagraphs_if_requested(torch::Device device) {
    if (!device.is_cuda())
        return;
    if (std::getenv("ALPHAZERO_ENABLE_CUDAGRAPHS") == nullptr)
        return;
    constexpr int64_t kSubgraphCudagraphs = 1; // torch_tensorrt::core::runtime::SUBGRAPH_CUDAGRAPHS
    auto op = c10::Dispatcher::singleton().findSchema({"tensorrt::set_cudagraphs_mode", ""});
    if (!op.has_value()) {
        spdlog::warn("tensorrt::set_cudagraphs_mode op not found; CUDAGraphs mode not enabled "
                    "(is libtorchtrt.so actually loaded?)");
        return;
    }
    op->typed<void(int64_t)>().call(kSubgraphCudagraphs);
    spdlog::info("CUDAGraphs mode enabled for TensorRT inference (ALPHAZERO_ENABLE_CUDAGRAPHS set)");
}
#endif

NetworkInfererFactory::NetworkInfererFactory(std::string network_file_path, torch::Device device,
                                             int wait_for_count, int timeout_ms)
    : network_file_path(std::move(network_file_path)), device(device),
      wait_for_count(wait_for_count), timeout_ms(timeout_ms),
      network(get_network_func(this->network_file_path, device)) {
    network->to(device);
    network->eval();
#ifdef ALPHAZERO_TRT_LIB_PATH
    enable_cudagraphs_if_requested(device);
#endif
    batcher = std::make_shared<DynamicBatcher>(wait_for_count, timeout_ms, network, device);
}

std::unique_ptr<Inferer> NetworkInfererFactory::get_inferer() {
    auto lock_guard = std::scoped_lock(get_inferer_mutex);
    return std::make_unique<NetworkInferer>(batcher, device);
}
