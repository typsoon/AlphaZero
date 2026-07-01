#include "basic_inferer.hpp"
#include "inferer.hpp"
#include <ATen/core/TensorBody.h>
#include <condition_variable>
#include <connect4.hpp>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
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
    std::vector<int64_t> single_shape;

    std::vector<std::shared_ptr<Task>> pending_tasks;
    int current_count = 0;

    std::shared_ptr<Network> network;
    torch::jit::Method infer_method;
    torch::Device device;

    std::vector<inference_result> execute_tensor_batch(torch::Tensor batched) {
        if (batched.size(0) == 0)
            return {};
        auto batch_size = batched.size(0);
        batched = batched.to(device);

        std::vector<inference_result> out;
        try {
            torch::NoGradGuard no_grad;
            auto result = infer_method({batched});
            auto outputs = result.toTuple()->elements();
            torch::Tensor policy = outputs[0].toTensor().cpu().contiguous();
            torch::Tensor value = outputs[1].toTensor().cpu().contiguous();
            const float *value_ptr = value.data_ptr<float>();

            out.reserve(batch_size);
            for (int64_t i = 0; i < batch_size; ++i) {
                out.push_back(inference_result{policy, static_cast<int>(i), value_ptr[i]});
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

        std::vector<inference_result> results;
        try {
            results = execute_tensor_batch(giant_batch);
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

NetworkInfererFactory::NetworkInfererFactory(std::string network_file_path, torch::Device device,
                                             int wait_for_count, int timeout_ms)
    : network_file_path(std::move(network_file_path)), device(device),
      wait_for_count(wait_for_count), timeout_ms(timeout_ms),
      network(get_network_func(this->network_file_path, device)) {
    network->to(device);
    network->eval();
    batcher = std::make_shared<DynamicBatcher>(wait_for_count, timeout_ms, network, device);
}

std::unique_ptr<Inferer> NetworkInfererFactory::get_inferer() {
    auto lock_guard = std::scoped_lock(get_inferer_mutex);
    return std::make_unique<NetworkInferer>(batcher, device);
}
