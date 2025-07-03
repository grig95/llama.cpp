#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "arg.h"
#include "common.h"
#include "ggml.h"
#include "llama-arch.h"
#include "llama-model.h"
#include "llama.h"
#include "log.h"

// output file
static std::ofstream out_csv;

static llama_model* g_model;

/**
 * This the arbitrary data which will be passed to each callback.
 * Later on we can for example add operation or tensor name filter from the CLI arg, or a file descriptor to dump the tensor.
 */
struct callback_data {
    std::vector<uint8_t> data;
};

static std::string ggml_ne_string(int64_t const* ne) {
    std::string str;
    str += "{";
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    str += "}";
    return str;
}

static std::string ggml_ne_string(const ggml_tensor * t) {
    return ggml_ne_string(t->ne);
}

/**
 *  @param csv output stream to output to in csv format
 *  @param data tensor's data, note that this MUST be a copy of t->data on memory accessible to the cpu
 *  if t is on gpu or other hardware
 *  @param id id of the eval_callback call
 *  @param t pointer to tensor, used to get type, ne and nb (WON'T ACCESS t->data, use data param for that)
 */
static void ggml_print_tensor_to_csv(std::ostream& csv, uint8_t * data, size_t id, ggml_tensor const* t) {
    ggml_type type = t->type;
    const int64_t * ne = t->ne;
    const size_t * nb = t->nb;
    float* unquantized_data = nullptr;

    if (ggml_is_quantized(t->type)) {
        unquantized_data = new float[ggml_nelements(t)];
        ggml_get_type_traits(t->type)->to_float(data, unquantized_data, ggml_nelements(t));
    }

    int64_t curr_ne[GGML_MAX_DIMS];
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v;

                    if (!unquantized_data) {
                        if (type == GGML_TYPE_F16) {
                            v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[i]);
                        } else if (type == GGML_TYPE_F32) {
                            v = *(float *) &data[i];
                        } else if (type == GGML_TYPE_I32) {
                            v = (float) *(int32_t *) &data[i];
                        } else if (type == GGML_TYPE_I16) {
                            v = (float) *(int16_t *) &data[i];
                        } else if (type == GGML_TYPE_I8) {
                            v = (float) *(int8_t *) &data[i];
                        } else {
                            GGML_ABORT("fatal error");
                        }
                    }
                    else {
                        v = unquantized_data[i];
                    }

                    // print
                    // format: ID,name,type,operation,full_ne,curr_ne,value (see init_csv_columns)
                    curr_ne[0]=i0;
                    curr_ne[1]=i1;
                    curr_ne[2]=i2;
                    curr_ne[3]=i3;
                    csv<<id<<',';
                    csv<<t->name<<',';
                    csv<<ggml_type_name(type)<<','<<ggml_op_desc(t)<<',';
                    csv<<'"'<<ggml_ne_string(ne)<<"\",";
                    csv<<'"'<<ggml_ne_string(curr_ne)<<"\",";
                    csv<<v<<'\n';
                }
            }
        }
    }

    if (unquantized_data)
        delete[] unquantized_data;
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
static bool ggml_eval_layer_output(struct ggml_tensor * t, bool ask, void * user_data) {
    static size_t id = 0;
    auto * cb_data = (callback_data *) user_data;

    if (ask) {
        if (std::strncmp(t->name, "l_out", 5) == 0)
            return true;
        if (g_model->arch == LLM_ARCH_WAVTOKENIZER_DEC
            && (std::strncmp(t->name, "posnet_out", 10) == 0
                || std::strncmp(t->name, "convnext_out", 12)))
            return true;
        return false;
    }

    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    if(!ggml_is_quantized(t->type)) {
        uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
        ggml_print_tensor_to_csv(out_csv, data, id, t);
    }

    id++;

    return true;
}

static void init_csv_columns(std::ostream& csv) {
    csv<<"ID,name,type,operation,full_ne,curr_ne,value\n";
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    callback_data cb_data;

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_HIDDEN_STATES)) {
        return 1;
    }

    out_csv.open(params.out_file.c_str());
    init_csv_columns(out_csv);

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ggml_eval_layer_output;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    // init
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // global model needed for architecture check
    g_model = model;

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
