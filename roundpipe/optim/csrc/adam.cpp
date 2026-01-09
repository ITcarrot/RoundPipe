#include <omp.h>
#include <torch/extension.h>
using namespace std;
using namespace torch;

template <bool amsgrad, bool maximize, bool zero_weight_decay,
          bool decoupled_weight_decay>
void adam_kernel(float *__restrict params, const float *__restrict grads,
                 float *__restrict exp_avg, float *__restrict exp_avg_sq,
                 float *__restrict max_exp_avg_sq, float lr, float beta1, float beta2,
                 float eps, float weight_decay, int64_t param_size, int64_t step) {
    float one_beta1 = 1.0f - beta1;
    float one_beta2 = 1.0f - beta2;
    float bias_correction1 = 1.0f - powf(beta1, step);
    float bias_correction2 = 1.0f - powf(beta2, step);
    float one_lr_weight_decay = 1.0f - lr * weight_decay;
    float step_size = lr / bias_correction1;
    float div_bias_correction2 = 1.0f / bias_correction2;

    for (int64_t i = 0; i < param_size; ++i) {
        float grad = !maximize ? grads[i] : -grads[i];
        if (!zero_weight_decay) {
            if (decoupled_weight_decay) {
                params[i] *= one_lr_weight_decay;
            } else {
                grad += weight_decay * params[i];
            }
        }
        exp_avg[i] = beta1 * exp_avg[i] + one_beta1 * grad;
        exp_avg_sq[i] = beta2 * exp_avg_sq[i] + one_beta2 * grad * grad;
        float denom;
        if (amsgrad) {
            max_exp_avg_sq[i] = fmaxf(max_exp_avg_sq[i], exp_avg_sq[i]);
            denom = sqrtf(max_exp_avg_sq[i] * div_bias_correction2) + eps;
        } else {
            denom = sqrtf(exp_avg_sq[i] * div_bias_correction2) + eps;
        }
        params[i] -= step_size * exp_avg[i] / denom;
    }
}

// Helper to unpack boolean template parameters
template <bool... FixedBools, typename... Args>
void adam_kernel(bool current_bool, Args... args) {
    if (current_bool) {
        adam_kernel<FixedBools..., true>(args...);
    } else {
        adam_kernel<FixedBools..., false>(args...);
    }
}

void adam(vector<Tensor> params, vector<Tensor> grads, vector<Tensor> exp_avg,
          vector<Tensor> exp_avg_sq, vector<Tensor> max_exp_avg_sq,
          vector<int64_t> step_int, bool amsgrad, float beta1, float beta2, float lr,
          float weight_decay, float eps, bool maximize, bool decoupled_weight_decay) {
    vector<int64_t> numel(params.size());
    vector<float *> params_ptr(params.size());
    vector<const float *> grads_ptr(params.size());
    vector<float *> exp_avg_ptr(params.size());
    vector<float *> exp_avg_sq_ptr(params.size());
    vector<float *> max_exp_avg_sq_ptr(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        numel[i] = params[i].numel();
        params_ptr[i] = params[i].mutable_data_ptr<float>();
        grads_ptr[i] = grads[i].const_data_ptr<float>();
        exp_avg_ptr[i] = exp_avg[i].mutable_data_ptr<float>();
        exp_avg_sq_ptr[i] = exp_avg_sq[i].mutable_data_ptr<float>();
        if (amsgrad) {
            max_exp_avg_sq_ptr[i] = max_exp_avg_sq[i].mutable_data_ptr<float>();
        } else {
            max_exp_avg_sq_ptr[i] = nullptr;
        }
    }
#pragma omp parallel
    {
        int rank = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        for (size_t i = 0; i < params.size(); ++i) {
            int64_t block_size = numel[i] / nthreads + (rank < (numel[i] % nthreads));
            int64_t offset =
                (numel[i] / nthreads) * rank + min<int64_t>(rank, numel[i] % nthreads);
            adam_kernel(amsgrad, maximize, weight_decay == 0.0f, decoupled_weight_decay,
                        params_ptr[i] + offset, grads_ptr[i] + offset,
                        exp_avg_ptr[i] + offset, exp_avg_sq_ptr[i] + offset,
                        max_exp_avg_sq_ptr[i] + offset, lr, beta1, beta2, eps,
                        weight_decay, block_size, step_int[i]);
        }
    }
}

#if PYBIND11_VERSION_HEX >= 0x020D0000 // pybind11 >= 2.13
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m, py::mod_gil_not_used())
#else
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
#endif
{
    m.def("adam", &adam, py::call_guard<py::gil_scoped_release>(),
          "Adam optimizer step implementation in C++");
}
