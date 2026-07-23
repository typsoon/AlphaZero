#include "basic_inferer.hpp"
#include "encoder_factory.hpp"
#include "inference_cache.hpp"
#include "inferer.hpp"
#include <ATen/core/TensorBody.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <condition_variable>
#include <connect4.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <spdlog/spdlog.h>
#include <thread>
#include <unordered_map>
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
    // Two threads, pipelined: `worker` gathers submitted tasks (as before) and then
    // does the CPU-side prep for a batch (write_canonical_state, legal-action
    // extraction) into one of two alternating buffer slots; `gpu_executor_thread`
    // takes each prepared batch and does the GPU-side work (H2D copy, forward pass,
    // gather, D2H, promise fulfillment). Splitting these into separate threads lets
    // `worker` start gathering/prepping the *next* batch (into the other slot)
    // while `gpu_executor_thread` is still working on the current one, instead of
    // the single worker thread doing both in sequence - measured (via perf sched)
    // to matter because process_batch's state-write loop is entirely single-
    // threaded and was sitting squarely in between every GPU launch.
    std::thread worker;
    std::thread gpu_executor_thread;

    struct Task {
        std::vector<const GameState *> states;
        int count{};
        std::promise<std::vector<inference_result>> promise;
    };

    struct BufferSlot {
        torch::Tensor pinned_buffer;
        torch::Tensor pinned_index_buffer;
        torch::Tensor pinned_gathered_buffer;
        torch::Tensor pinned_value_buffer;
    };
    // Scale-dependent: at thread_count=12 (this dev machine's real core count),
    // kineto A/B testing (60 games/mcts_batch_size=64) found kNumBufferSlots=2 vs.
    // 3 made no measurable difference (~20% GPU idle either way) - with only 2 real
    // pipeline stages (worker's CPU prep, gpu_executor_thread's GPU work), 2 slots
    // already let worker run a full batch ahead, which seemed like enough to fully
    // overlap the two stages. But re-tested at thread_count=50 (matching
    // chess_params.json's production setting, 100 games): kNumBufferSlots=3 is
    // reproducibly ~16-17% faster wall-clock (37.2-37.6s vs. 44.1-44.4s at 2,
    // repeated twice each). GPU idle *ratio* stays similar between the two there
    // too, but total GPU busy time itself shrinks at 3 slots (35.4s -> ~29.4s) -
    // consistent with the extra slot letting worker get further ahead under
    // thread_count=50's much larger batch sizes (measured median 336, up to 2244 -
    // see training/self_play.cpp git history), forming fewer/larger aggregate
    // batches per gpu_executor_thread call and cutting per-call overhead, not just
    // hiding CPU-prep time behind GPU exec. Kept at 3: it's neutral-to-better at
    // low thread counts and a real win at the production-scale ones.
    static constexpr int kNumBufferSlots = 3;
    BufferSlot buffer_slots[kNumBufferSlots];
    std::vector<int64_t> single_shape;
    // Encodes states into the batch tensor. Null until resolved from the game
    // type on the first prepare_batch (or supplied by the factory). The worker
    // thread is the only one that touches it, so lazy init needs no lock.
    std::shared_ptr<StateEncoder> encoder;

    struct PreparedBatch {
        std::vector<std::shared_ptr<Task>> tasks;
        int slot{};
        int games_count{};
        int total_count{};
        std::vector<int> legal_actions_flat;
        std::vector<int> legal_actions_offsets;
    };

    // Handoff between worker and gpu_executor_thread. slot_busy[i] is true from the
    // moment worker hands off a batch using slot i until gpu_executor_thread
    // finishes all GPU work and promise fulfillment for it - worker must wait for
    // slot_busy[i] to clear before it can start writing a new batch into slot i.
    // ready_batch is a capacity-1 handoff (worker sets it, gpu_executor_thread
    // takes it) guarded by the same mutex/cv.
    std::mutex pipeline_mtx;
    std::condition_variable pipeline_cv;
    bool slot_busy[kNumBufferSlots] = {};
    std::optional<PreparedBatch> ready_batch;
    bool gpu_stop = false;

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
                         const std::vector<int> &legal_actions_offsets, int slot) {
        if (batched.size(0) == 0)
            return {};
        auto batch_size = batched.size(0);
        batched = batched.to(device);
        BufferSlot &buf = buffer_slots[slot];

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
                copy_stream =
                    c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
            }
            std::optional<c10::cuda::CUDAStreamGuard> stream_guard;
            if (device.is_cuda()) {
                // copy_stream is always populated above whenever device.is_cuda().
                stream_guard.emplace(*copy_stream); // NOLINT(bugprone-unchecked-optional-access)
            }

            auto result = infer_method({batched});
            auto outputs = result.toTuple()->elements();
            // .to(device) is a no-op for a well-behaved model whose outputs already
            // match the input's device. It's a real fixup for models that internally
            // construct fresh tensors without a device= arg (e.g. torch.ones(...)),
            // which silently land on the CPU default regardless of the input device -
            // without this, the gather() calls below would mismatch devices.
            auto policy_gpu = outputs[0].toTensor().to(device);
            auto value_gpu = outputs[1].toTensor().to(device);

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
                if (!buf.pinned_value_buffer.defined() ||
                    buf.pinned_value_buffer.size(0) < batch_size) {
                    auto value_shape = value_gpu.sizes().vec();
                    value_shape[0] =
                        std::max<int64_t>(batch_size, static_cast<int64_t>(wait_for_count) * 2);
                    buf.pinned_value_buffer = torch::empty(
                        value_shape,
                        torch::TensorOptions().dtype(value_gpu.dtype()).pinned_memory(true));
                }
                if (max_actions > 0) {
                    if (!buf.pinned_index_buffer.defined() ||
                        buf.pinned_index_buffer.size(0) < batch_size ||
                        buf.pinned_index_buffer.size(1) < max_actions) {
                        buf.pinned_index_buffer = torch::zeros(
                            {std::max<int64_t>(batch_size,
                                               static_cast<int64_t>(wait_for_count) * 2),
                             max_actions},
                            torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
                        buf.pinned_gathered_buffer = torch::empty(
                            {std::max<int64_t>(batch_size,
                                               static_cast<int64_t>(wait_for_count) * 2),
                             max_actions},
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
                    gather_width = buf.pinned_gathered_buffer.size(1);

                    auto index_host = buf.pinned_index_buffer.slice(0, 0, batch_size);
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

                    // TEMP DIAGNOSTIC: validate every index before it reaches the GPU
                    // gather() call, to pin down the exact bad value instead of a bare
                    // CUDA device-side assert.
                    {
                        int64_t action_dim = policy_gpu.size(1);
                        for (int64_t i = 0; i < batch_size; ++i) {
                            int begin = legal_actions_offsets[i];
                            int end = legal_actions_offsets[i + 1];
                            for (int j = begin; j < end; ++j) {
                                int action = legal_actions_flat[j];
                                if (action < 0 || action >= action_dim) {
                                    spdlog::error("BAD ACTION INDEX: row={} j={} action={} "
                                                  "action_dim={} begin={} end={} "
                                                  "row_len={} batch_size={} "
                                                  "legal_actions_flat.size()={} "
                                                  "legal_actions_offsets.size()={}",
                                                  i, j, action, action_dim, begin, end, end - begin,
                                                  batch_size, legal_actions_flat.size(),
                                                  legal_actions_offsets.size());
                                }
                            }
                        }
                    }

                    // Stream-ordered on the same (default) CUDA stream as the H2D copy
                    // below and the forward pass above, so no manual sync is needed
                    // between them - CUDA guarantees ordering within a stream.
                    auto index_gpu = index_host.to(device, true);
                    auto gathered_gpu = policy_gpu.gather(1, index_gpu);

                    auto gathered_dst = buf.pinned_gathered_buffer.slice(0, 0, batch_size);
                    gathered_dst.copy_(gathered_gpu, true);
                    gathered_host = gathered_dst;
                }

                auto value_dst = buf.pinned_value_buffer.slice(0, 0, batch_size);
                value_dst.copy_(value_gpu, true);
                // Everything above (forward pass, gather, both D2H copies) ran on
                // copy_stream, so waiting on it alone - rather than the whole device -
                // is sufficient and avoids blocking on unrelated device activity.
                copy_stream->synchronize(); // NOLINT(bugprone-unchecked-optional-access)

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

    // CPU-only: writes canonical states + extracts legal actions into buffer_slots
    // [slot]. No GPU calls here at all - that's the whole point of splitting this
    // out from what used to be process_batch(), so it can run on `worker` while
    // `gpu_executor_thread` is still busy with a previous batch in the other slot.
    std::optional<PreparedBatch> prepare_batch(std::vector<std::shared_ptr<Task>> tasks, int slot) {
        int total_count = 0;
        int games_count = 0;
        for (const auto &t : tasks) {
            total_count += t->count;
            games_count += t->count;
        }
        if (total_count == 0)
            return std::nullopt;

        if (!encoder) {
            for (const auto &t : tasks) {
                if (!t->states.empty()) {
                    encoder = default_encoder_for(*t->states[0]);
                    break;
                }
            }
        }
        if (single_shape.empty() && encoder) {
            single_shape = encoder->state_shape();
        }

        BufferSlot &buf = buffer_slots[slot];
        if (games_count > 0) {
            if (!buf.pinned_buffer.defined() || buf.pinned_buffer.size(0) < games_count) {
                auto alloc_shape = single_shape;
                alloc_shape.insert(alloc_shape.begin(), std::max(games_count, wait_for_count * 2));
                // Pinned (page-locked) host memory only accelerates the async
                // H2D copy to a CUDA device; on a CPU device it buys nothing and
                // - worse - forces a CUDA context init (the pinned allocator is
                // CUDA's host allocator), which fails outright when no GPU is
                // usable (e.g. the CPU-only sanitizer stress harnesses, or a box
                // whose GPU is fully occupied). Only request it on the CUDA path.
                auto options =
                    torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(device.is_cuda());
                buf.pinned_buffer = torch::empty(alloc_shape, options);
            }

            auto *giant_data = buf.pinned_buffer.data_ptr<float>();
            int state_size = 1;
            for (long i : single_shape)
                state_size *= i;

            int offset = 0;
            for (const auto &t : tasks) {
                for (auto &state : t->states) {
                    encoder->write_canonical_state(
                        *state, giant_data + (static_cast<ptrdiff_t>(offset * state_size)));
                    offset++;
                }
            }
        }

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

        PreparedBatch pb;
        pb.tasks = std::move(tasks);
        pb.slot = slot;
        pb.games_count = games_count;
        pb.total_count = total_count;
        pb.legal_actions_flat = std::move(legal_actions_flat);
        pb.legal_actions_offsets = std::move(legal_actions_offsets);
        return pb;
    }

    // GPU-only (plus the H2D copy that kicks it off): runs on gpu_executor_thread.
    // Frees pb.slot (via slot_busy) once done, letting `worker` reuse it for a
    // future batch.
    void execute_prepared_batch(PreparedBatch pb) {
        BufferSlot &buf = buffer_slots[pb.slot];
        torch::Tensor giant_batch;
        if (pb.games_count > 0) {
            giant_batch =
                buf.pinned_buffer.slice(0, 0, pb.games_count).to(device, /*non_blocking=*/true);
        }

        if (device.is_cuda()) {
            // slot_busy[pb.slot] exists to protect exactly one thing from the
            // cross-thread race worker_loop()'s wait(!slot_busy[slot]) guards
            // against: buf.pinned_buffer, the only per-slot buffer prepare_batch()
            // (called from worker_loop, a different thread) writes into - the
            // other three BufferSlot members (pinned_index_buffer,
            // pinned_gathered_buffer, pinned_value_buffer) are written only by
            // execute_tensor_batch() below, itself, on this same gpu_executor_thread,
            // serially, so they need no cross-thread protection at all.
            // buf.pinned_buffer's *only* reader is the H2D copy just above - so
            // once that copy has genuinely completed, the slot is already safe to
            // reuse, without waiting for the rest of this function (forward pass,
            // gather, D2H) to finish too.
            //
            // The synchronize() below is required to make that true: with
            // non_blocking=true, .to(device) only *issues* the copy and returns
            // immediately - it does not wait for it. execute_tensor_batch()'s own
            // copy_stream->synchronize() (further down) does not cover this either,
            // since it only waits for copy_stream's own work (forward-pass
            // consumption + gather + D2H); it does not wait for this H2D copy
            // (issued on whatever stream was ambient at this call site, before
            // copy_stream's guard is even entered), nor for TensorRT's own
            // execution - measured via nsys (cuda_gpu_trace on a live self-play
            // run): TensorRT's kernels consistently land on their own separate
            // internal stream, distinct from both the H2D-copy stream and
            // copy_stream, with no observed dependency linking any of the three,
            // and 203 confirmed cases were found where TensorRT's first kernel
            // (which reads this same giant_batch data) started executing *while*
            // this H2D copy was still in flight. Without an explicit sync here,
            // clearing slot_busy[pb.slot] (previously done only at the very end of
            // this function, after copy_stream->synchronize()) was not actually a
            // reliable signal that buf.pinned_buffer was safe to overwrite -
            // reopening the same host-memory race the slot_busy wait in
            // worker_loop() exists to prevent, just via a path the CPU-side
            // pipeline_mtx/slot_busy bookkeeping alone couldn't see. A full device
            // sync (rather than something scoped to just the specific streams
            // involved) sidesteps needing to pin down every stream TensorRT's
            // runtime might use internally.
            c10::cuda::device_synchronize();
        }

        {
            std::scoped_lock plock(pipeline_mtx);
            slot_busy[pb.slot] = false;
        }
        pipeline_cv.notify_all();

        std::vector<inference_result> results;
        std::exception_ptr ex;
        try {
            results = execute_tensor_batch(giant_batch, pb.legal_actions_flat,
                                           pb.legal_actions_offsets, pb.slot);
        } catch (...) {
            ex = std::current_exception();
        }

        if (ex) {
            for (auto &t : pb.tasks) {
                t->promise.set_exception(ex);
            }
        } else {
            int offset = 0;
            for (auto &t : pb.tasks) {
                std::vector<inference_result> chunk;
                chunk.reserve(t->count);
                for (int i = 0; i < t->count; i++) {
                    chunk.push_back(results[offset++]);
                }
                t->promise.set_value(std::move(chunk));
            }
        }
    }

    void worker_loop() {
        int next_slot = 0;
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

            if (tasks_to_execute.empty())
                continue;

            int slot = next_slot;
            next_slot = (next_slot + 1) % kNumBufferSlots;

            // Wait for this specific slot to be free (i.e. gpu_executor_thread
            // finished the batch that last used it, two rounds ago) BEFORE
            // prepare_batch() below writes into buffer_slots[slot]'s pinned
            // memory - not just before the ready_batch handoff. prepare_batch()
            // writes canonical states directly into that slot's pinned_buffer,
            // and gpu_executor_thread's execute_prepared_batch() reads from
            // that same buffer via a non-blocking H2D copy that can still be
            // in flight; writing into it here first (the previous bug) raced
            // that in-flight read/copy, corrupting the batch gpu_executor_thread
            // was still processing and, depending on allocator behavior when
            // the buffer needed to grow, unrelated heap memory too (observed
            // as a glibc `_int_malloc` heap-corruption abort during self-play).
            {
                std::unique_lock<std::mutex> plock(pipeline_mtx);
                pipeline_cv.wait(plock, [&] { return !slot_busy[slot]; });
            }

            auto prepared = prepare_batch(std::move(tasks_to_execute), slot);
            if (!prepared)
                continue;

            // Separate wait for the single-item ready_batch handoff to be
            // clear (gpu_executor_thread picked up whatever was there before) -
            // independent of the slot-readiness wait above, since a free slot
            // doesn't imply the handoff itself is free.
            std::unique_lock<std::mutex> plock(pipeline_mtx);
            pipeline_cv.wait(plock, [&] { return !ready_batch.has_value(); });
            slot_busy[slot] = true;
            ready_batch = std::move(*prepared);
            plock.unlock();
            pipeline_cv.notify_all();
        }

        {
            std::scoped_lock plock(pipeline_mtx);
            gpu_stop = true;
        }
        pipeline_cv.notify_all();
    }

    void gpu_executor_loop() {
        while (true) {
            PreparedBatch pb;
            {
                std::unique_lock<std::mutex> plock(pipeline_mtx);
                pipeline_cv.wait(plock, [&] { return gpu_stop || ready_batch.has_value(); });
                if (!ready_batch.has_value()) {
                    if (gpu_stop)
                        break;
                    continue;
                }
                pb = std::move(*ready_batch);
                ready_batch.reset();
            }
            // Let `worker` know the handoff slot is free before doing the (slow) GPU
            // work below, instead of after - worker can start preparing its next
            // batch immediately rather than waiting on this round's GPU work too.
            pipeline_cv.notify_all();

            execute_prepared_batch(std::move(pb));
        }
    }

  public:
    DynamicBatcher(int wait_for_count, int timeout_ms, std::shared_ptr<Network> network,
                   torch::Device device, std::shared_ptr<StateEncoder> encoder = nullptr)
        : wait_for_count(wait_for_count), timeout_ms(timeout_ms), network(network),
          infer_method(network->get_method("forward")), device(device) {
        this->encoder = std::move(encoder);
        worker = std::thread(&DynamicBatcher::worker_loop, this);
        gpu_executor_thread = std::thread(&DynamicBatcher::gpu_executor_loop, this);
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
        if (gpu_executor_thread.joinable()) {
            gpu_executor_thread.join();
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

NetworkInferer::NetworkInferer(std::shared_ptr<DynamicBatcher> batcher, torch::Device device,
                               std::shared_ptr<InferenceCache> cache,
                               std::shared_ptr<StateEncoder> encoder)
    : Inferer(device), batcher(std::move(batcher)), cache(std::move(cache)),
      encoder(std::move(encoder)) {}

vector<inference_result> NetworkInferer::infer(const vector<const GameState *> &states) {
    if (states.empty())
        return {};

    if (!cache)
        return batcher->submit(states);

    if (!encoder)
        encoder = default_encoder_for(*states[0]);

    size_t n = states.size();
    vector<inference_result> results(n);
    std::vector<uint64_t> keys(n);
    std::vector<bool> hit(n, false);

    // The cache key must be the exact network input, so the canonical tensor
    // is written once into a per-thread scratch buffer here just to be hashed
    // (see inference_cache.hpp for why no game-level hash is safe). For
    // misses that tensor write happens again inside the batcher's
    // prepare_batch - redundant, but tiny next to the inference the hits are
    // saving.
    static thread_local std::vector<float> scratch;
    auto shape = encoder->state_shape();
    size_t state_size = 1;
    for (int64_t d : shape)
        state_size *= static_cast<size_t>(d);
    for (size_t i = 0; i < n; ++i) {
        scratch.resize(state_size);
        encoder->write_canonical_state(*states[i], scratch.data());
        keys[i] = InferenceCache::hash_state(scratch.data(), state_size);
        hit[i] = cache->lookup(keys[i], results[i]);
    }

    // Submit only the misses, deduplicated by key: distinct MCTS nodes in one
    // batch can still be transpositions of each other (the caller's own
    // dedupe in evaluate_batch() is by Node*, which can't see that), and
    // identical inputs would produce identical outputs anyway.
    std::vector<const GameState *> miss_states;
    std::vector<size_t> miss_index_of(n, 0);
    std::unordered_map<uint64_t, size_t> first_miss_with_key;
    for (size_t i = 0; i < n; ++i) {
        if (hit[i])
            continue;
        auto [it, inserted] = first_miss_with_key.try_emplace(keys[i], miss_states.size());
        miss_index_of[i] = it->second;
        if (inserted)
            miss_states.push_back(states[i]);
    }

    if (!miss_states.empty()) {
        auto miss_results = batcher->submit(miss_states);
        for (const auto &[key, miss_idx] : first_miss_with_key) {
            cache->insert(key, miss_results[miss_idx]);
        }
        // Copy, not move: several i's can share one deduplicated miss_results
        // element, and a move would gut it for every consumer after the first.
        for (size_t i = 0; i < n; ++i) {
            if (!hit[i])
                results[i] = miss_results[miss_index_of[i]];
        }
    }

    // TEMP DIAGNOSTIC (env-gated): verify every delivered result actually
    // belongs to the state it's paired with, by comparing its legal_actions
    // against a fresh recomputation. Distinguishes cache-served poison from
    // batcher-level mispairing at the exact point of delivery.
    static const bool verify_results = std::getenv("ALPHAZERO_VERIFY_INFER_RESULTS") != nullptr;
    if (verify_results) {
        for (size_t i = 0; i < n; ++i) {
            auto expected = states[i]->get_legal_actions();
            if (results[i].legal_actions != expected) {
                std::string got;
                for (int a : results[i].legal_actions)
                    got += std::to_string(a) + " ";
                std::string want;
                for (int a : expected)
                    want += std::to_string(a) + " ";
                spdlog::critical("INFER RESULT MISMATCH: i={} source={} key={:#x} "
                                 "got_legal_actions=[{}] expected=[{}]",
                                 i, hit[i] ? "CACHE_HIT" : "BATCHER_MISS", keys[i], got, want);
                std::abort();
            }
        }
    }

    return results;
}

static std::shared_ptr<Network> get_network_func(std::string network_file_path,
                                                 torch::Device device) {
    if (std::filesystem::exists(network_file_path)) {
        try {
#ifdef ALPHAZERO_TRT_LIB_PATH
            static const bool trt_runtime_loaded = []() {
                void *handle = dlopen(ALPHAZERO_TRT_LIB_PATH, RTLD_NOW | RTLD_GLOBAL);
                if (handle == nullptr) {
                    const char *dlopen_error = dlerror(); // NOLINT(concurrency-mt-unsafe)
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
    if (std::getenv("ALPHAZERO_ENABLE_CUDAGRAPHS") == nullptr) // NOLINT(concurrency-mt-unsafe)
        return;
    constexpr int64_t kSubgraphCudagraphs = 1; // torch_tensorrt::core::runtime::SUBGRAPH_CUDAGRAPHS
    auto op = c10::Dispatcher::singleton().findSchema({"tensorrt::set_cudagraphs_mode", ""});
    if (!op.has_value()) {
        spdlog::warn("tensorrt::set_cudagraphs_mode op not found; CUDAGraphs mode not enabled "
                     "(is libtorchtrt.so actually loaded?)");
        return;
    }
    op->typed<void(int64_t)>().call(kSubgraphCudagraphs);
    spdlog::info(
        "CUDAGraphs mode enabled for TensorRT inference (ALPHAZERO_ENABLE_CUDAGRAPHS set)");
}
#endif

NetworkInfererFactory::NetworkInfererFactory(std::string network_file_path, torch::Device device,
                                             int wait_for_count, int timeout_ms,
                                             size_t transposition_cache_entries,
                                             std::shared_ptr<StateEncoder> encoder)
    : network_file_path(std::move(network_file_path)), device(device),
      wait_for_count(wait_for_count), timeout_ms(timeout_ms),
      network(get_network_func(this->network_file_path, device)) {
    this->encoder = std::move(encoder);
    network->to(device);
    network->eval();
#ifdef ALPHAZERO_TRT_LIB_PATH
    enable_cudagraphs_if_requested(device);
#endif
    batcher = std::make_shared<DynamicBatcher>(wait_for_count, timeout_ms, network, device,
                                               this->encoder);
    if (transposition_cache_entries > 0) {
        cache = std::make_shared<InferenceCache>(transposition_cache_entries);
    }
}

NetworkInfererFactory::~NetworkInfererFactory() {
    if (cache) {
        uint64_t hits = cache->hits();
        uint64_t misses = cache->misses();
        uint64_t total = hits + misses;
        spdlog::info("Inference cache: {} hits / {} lookups ({:.1f}% hit rate)", hits, total,
                     total > 0 ? 100.0 * static_cast<double>(hits) / static_cast<double>(total)
                               : 0.0);
    }
}

std::unique_ptr<Inferer> NetworkInfererFactory::get_inferer() {
    auto lock_guard = std::scoped_lock(get_inferer_mutex);
    return std::make_unique<NetworkInferer>(batcher, device, cache, encoder);
}
