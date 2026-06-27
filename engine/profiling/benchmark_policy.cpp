#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <torch/torch.h>
#include <vector>

std::vector<float> sample_dirichlet(const std::vector<float> &alpha) {
    static thread_local std::mt19937 gen{std::random_device{}()};
    std::vector<float> x(alpha.size());
    float sum = 0.0f;
    for (size_t i = 0; i < alpha.size(); ++i) {
        std::gamma_distribution<float> dist(alpha[i], 1.0f);
        x[i] = dist(gen);
        sum += x[i];
    }
    if (sum > 0.0f) {
        for (auto &v : x)
            v /= sum;
    }
    return x;
}

std::vector<float> get_policy_from_logits_aten(torch::Tensor policy_logits,
                                               const std::vector<int> &legal_actions,
                                               bool dirichletNoise, float alpha, float eps) {
    policy_logits = policy_logits.squeeze(0); // [A]
    int A = policy_logits.size(0);

    std::vector<float> mask_vec(A, -std::numeric_limits<float>::infinity());
    for (int a : legal_actions) {
        mask_vec[a] = 0.0f;
    }
    torch::Tensor mask = torch::tensor(mask_vec, policy_logits.options());
    policy_logits = policy_logits + mask;

    torch::Tensor policy = torch::softmax(policy_logits, 0); // [A]

    if (dirichletNoise) {
        std::vector<float> alpha_vec(legal_actions.size(), alpha);
        auto noise_vec = sample_dirichlet(alpha_vec);

        std::vector<float> full_noise(A, 0.0f);
        for (size_t i = 0; i < legal_actions.size(); ++i) {
            full_noise[legal_actions[i]] = noise_vec[i];
        }

        auto noise_t = torch::tensor(full_noise, policy.options());
        policy = ((1.0f - eps) * policy) + (eps * noise_t);
    }

    policy = policy.cpu();
    std::vector<float> policy_vec(A, 0.0f);
    std::memcpy(policy_vec.data(), policy.data_ptr<float>(), A * sizeof(float));

    return policy_vec;
}

std::vector<float> get_policy_from_logits_cpp(torch::Tensor policy_logits,
                                              const std::vector<int> &legal_actions,
                                              bool dirichletNoise, float alpha_val, float eps) {
    policy_logits = policy_logits.squeeze(0); // [A]
    int A = policy_logits.size(0);
    const float *logits_data = policy_logits.data_ptr<float>();

    std::vector<float> policy_vec(A, 0.0f);

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int a : legal_actions) {
        if (logits_data[a] > max_logit) {
            max_logit = logits_data[a];
        }
    }

    float sum_exp = 0.0f;
    for (int a : legal_actions) {
        policy_vec[a] = std::exp(logits_data[a] - max_logit);
        sum_exp += policy_vec[a];
    }

    if (sum_exp > 0.0f) {
        for (int a : legal_actions) {
            policy_vec[a] /= sum_exp;
        }
    }

    if (dirichletNoise) {
        std::vector<float> alpha_vec(legal_actions.size(), alpha_val);
        auto noise_vec = sample_dirichlet(alpha_vec);

        for (size_t i = 0; i < legal_actions.size(); ++i) {
            int a = legal_actions[i];
            policy_vec[a] = ((1.0f - eps) * policy_vec[a]) + (eps * noise_vec[i]);
        }
    }

    return policy_vec;
}

int main() {
    int num_iterations = 100000;

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor logits = torch::randn({7}, options);
    std::vector<int> legal_actions = {0, 1, 2, 4, 6};

    // Warmup
    for (int i = 0; i < 100; i++) {
        get_policy_from_logits_aten(logits, legal_actions, false, 0.3f, 0.25f);
        get_policy_from_logits_cpp(logits, legal_actions, false, 0.3f, 0.25f);
    }

    float sum_aten_no = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto p = get_policy_from_logits_aten(logits, legal_actions, false, 0.3f, 0.25f);
        sum_aten_no += p[0];
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms_aten_no =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    float sum_cpp_no = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto p = get_policy_from_logits_cpp(logits, legal_actions, false, 0.3f, 0.25f);
        sum_cpp_no += p[0];
    }
    end = std::chrono::high_resolution_clock::now();
    double ms_cpp_no =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    float sum_aten_yes = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto p = get_policy_from_logits_aten(logits, legal_actions, true, 0.3f, 0.25f);
        sum_aten_yes += p[0];
    }
    end = std::chrono::high_resolution_clock::now();
    double ms_aten_yes =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    float sum_cpp_yes = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        auto p = get_policy_from_logits_cpp(logits, legal_actions, true, 0.3f, 0.25f);
        sum_cpp_yes += p[0];
    }
    end = std::chrono::high_resolution_clock::now();
    double ms_cpp_yes =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    std::cout << "Iterations: " << num_iterations << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << "No Dirichlet Noise:\n";
    std::cout << "ATen: " << ms_aten_no << " ms (" << (ms_aten_no * 1000 / num_iterations)
              << " us/call)\n";
    std::cout << "C++ : " << ms_cpp_no << " ms (" << (ms_cpp_no * 1000 / num_iterations)
              << " us/call)\n";
    std::cout << "Speedup: " << ms_aten_no / ms_cpp_no << "x\n";
    std::cout << "----------------------------------------\n";
    std::cout << "With Dirichlet Noise:\n";
    std::cout << "ATen: " << ms_aten_yes << " ms (" << (ms_aten_yes * 1000 / num_iterations)
              << " us/call)\n";
    std::cout << "C++ : " << ms_cpp_yes << " ms (" << (ms_cpp_yes * 1000 / num_iterations)
              << " us/call)\n";
    std::cout << "Speedup: " << ms_aten_yes / ms_cpp_yes << "x\n";

    // Prevent optimization
    if (sum_aten_no == -1.0)
        std::cout << sum_aten_no << sum_cpp_no << sum_aten_yes << sum_cpp_yes;

    return 0;
}
