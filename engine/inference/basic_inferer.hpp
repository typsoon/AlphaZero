#ifndef NETWORK_INFERER_HPP
#define NETWORK_INFERER_HPP

#include "game.hpp"
#include "inference_cache.hpp"
#include "inferer.hpp"
#include "state_encoder.hpp"
#include <connect4.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>

class DynamicBatcher;

class NetworkInferer : public Inferer {
  private:
    std::shared_ptr<DynamicBatcher> batcher;
    // Shared with every other NetworkInferer from the same factory; may be
    // null (caching disabled). See infer() for the lookup/miss-merge flow and
    // inference_cache.hpp for why results are keyed by canonical-tensor hash.
    std::shared_ptr<InferenceCache> cache;
    // Encodes states into the cache-key tensor. May be null: resolved lazily
    // from the game type on first use (see infer()), so callers that don't
    // specify an encoding keep working.
    std::shared_ptr<StateEncoder> encoder;

  public:
    NetworkInferer(std::shared_ptr<DynamicBatcher> batcher, torch::Device device,
                   std::shared_ptr<InferenceCache> cache = nullptr,
                   std::shared_ptr<StateEncoder> encoder = nullptr);

    vector<inference_result> infer(const vector<const GameState *> &states) override;
};

class NetworkInfererFactory : public InfererFactory {
  private:
    using Network = torch::jit::script::Module;

    std::string network_file_path;
    torch::Device device;
    int wait_for_count;
    int timeout_ms;

    std::shared_ptr<Network> network;
    std::shared_ptr<DynamicBatcher> batcher;
    // Owned here so its lifetime exactly matches this factory's network - a
    // cached (policy, value) is only valid for the weights that produced it,
    // and each self_play() invocation builds a fresh factory for the newly
    // trained checkpoint, so stale cross-iteration hits are impossible by
    // construction. Null when transposition_cache_entries == 0.
    std::shared_ptr<InferenceCache> cache;
    // The encoding this factory's network expects. Null = derive the default
    // from the game type at first use (see basic_infer.cpp); pass a specific
    // encoder to use a non-default encoding.
    std::shared_ptr<StateEncoder> encoder;

    std::mutex get_inferer_mutex;

  public:
    // transposition_cache_entries bounds the inference cache's slot count
    // (always-replace eviction above that); 0 disables caching entirely.
    NetworkInfererFactory(std::string network_file_path, torch::Device device,
                          int wait_for_count = 1, int timeout_ms = 10,
                          size_t transposition_cache_entries = 1000000,
                          std::shared_ptr<StateEncoder> encoder = nullptr);

    ~NetworkInfererFactory() override;

    std::unique_ptr<Inferer> get_inferer() override;
};

#endif // NETWORK_INFERER_HPP
