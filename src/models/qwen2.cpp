#include "models.h"

void llama_model_qwen2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_5B : LLM_TYPE_1B; break;
        case 28: type = hparams.n_embd == 1536 ? LLM_TYPE_1_5B : LLM_TYPE_7B; break;
        case 32: type = LLM_TYPE_7B; break;
        case 36: type = LLM_TYPE_3B; break;
        case 40: type = hparams.n_head() == 20 ? LLM_TYPE_4B : LLM_TYPE_13B; break;
        case 48: type = LLM_TYPE_14B; break;
        case 64: type = LLM_TYPE_32B; break;
        case 80: type = LLM_TYPE_70B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output_b    = create_tensor(tn(LLM_TENSOR_OUTPUT,      "bias"),   {n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen2::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int block_loops = get_recurrent_block_loops();
    auto [block_start, block_end] = get_recurrent_block_range(n_layer, model.arch, model.hparams.n_embd);

    auto build_layer = [&](int il, int bloop = 0, int bloops = 1, bool is_alt = false) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il, get_store_kv(bloop, bloops, is_alt));
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    };

    if (block_loops > 1 && block_start <= block_end) {
        // 1. Early syntactic layers (Zone 1)
        for (int il = 0; il < block_start; ++il) {
            build_layer(il);
        }

        // 2. Focal Macro-Reasoning Nexus (Zone 2: layers 12..19)
        // 2. Focal Macro-Reasoning Window
        ggml_tensor * block_inp_orig = inpL;
        ggml_tensor * first_pass_out = nullptr;

        for (int bloop = 0; bloop < block_loops; ++bloop) {
            for (int il = block_start; il <= block_end; ++il) {
                build_layer(il, bloop, block_loops, false);
            }
            if (bloop == 0) {
                first_pass_out = inpL;
            }
            if (bloop + 1 < block_loops) {
                float b_alpha = get_recurrent_block_alpha(bloop, block_loops, model.arch, model.hparams.n_embd);
                float momentum = get_recurrent_momentum();
                ggml_tensor * s_orig = ggml_scale(ctx0, block_inp_orig, 1.0f - b_alpha);
                ggml_tensor * s_cur  = ggml_scale(ctx0, inpL, b_alpha);
                inpL = ggml_add(ctx0, s_orig, s_cur);
                if (momentum > 0.0f) {
                    ggml_tensor * delta_m = ggml_sub(ctx0, s_cur, s_orig);
                    inpL = ggml_add(ctx0, inpL, ggml_scale(ctx0, delta_m, momentum));
                }
            }
        }

        // Exit Damping: blend refined state with anchor state h0 to stabilize logit variance
        if (first_pass_out != nullptr) {
            float exit_alpha = get_recurrent_block_exit_alpha(model.arch, model.hparams.n_embd, block_loops);
            if (exit_alpha < 1.0f) {
                ggml_tensor * s_pass1 = ggml_scale(ctx0, first_pass_out, 1.0f - exit_alpha);
                ggml_tensor * s_pass2 = ggml_scale(ctx0, inpL, exit_alpha);
                inpL = ggml_add(ctx0, s_pass1, s_pass2);
            }
        }

        // 3. Late logit calibration layers (Zone 3)
        for (int il = block_end + 1; il < n_layer; ++il) {
            build_layer(il);
        }
    } else {
        for (int il = 0; il < n_layer; ++il) {
            build_layer(il);
        }
    }

    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    if (model.output_b != nullptr) {
        cur = ggml_add(ctx0, cur, model.output_b);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
