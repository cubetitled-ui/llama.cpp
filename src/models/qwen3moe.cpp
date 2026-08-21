#include "models.h"

void llama_model_qwen3moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_30B_A3B; break;
        case 94: type = LLM_TYPE_235B_A22B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen3moe::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);

        if (n_expert == 0) {
            throw std::runtime_error("n_expert must be > 0 for QWEN3MOE");
        }
        if (n_expert_used == 0) {
            throw std::runtime_error("n_expert_used must be > 0 for QWEN3MOE");
        }

        // MoE branch
        const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen3moe::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen3moe::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
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
    auto [block_start, block_end] = get_recurrent_block_range(n_layer, model.arch);

    auto build_layer = [&](int il, int bloop = 0, int bloops = 1, bool is_alt = false) {
        if (bloop == 0 && !is_alt) {
            res->t_layer_inp[il] = inpL;
        }
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

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

        // MoE branch
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * moe_out =
            build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il,
                    nullptr, nullptr,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
        cb(moe_out, "ffn_moe_out", il);
        cur = moe_out;

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    };

    for (int il = 0; il < block_start; ++il) {
        build_layer(il, 0, 1, false);
    }

    if (block_loops > 1 && block_end >= block_start) {
        ggml_tensor * block_inp_orig = inpL;
        ggml_tensor * first_pass_out = nullptr;
        ggml_tensor * alt_stream_out = nullptr;
        const float alpha = get_recurrent_block_alpha(0, block_loops, model.arch);
        const float exit_alpha = get_recurrent_block_exit_alpha(model.arch, 0, block_loops);
        const bool dual_stream = get_recurrent_dual_stream();
        const float beta = get_recurrent_counter_beta();
        const float gamma = get_recurrent_counter_gamma();
        const float momentum = get_recurrent_momentum();

        for (int bloop = 0; bloop < block_loops; ++bloop) {
            for (int il = block_start; il <= block_end; ++il) {
                build_layer(il, bloop, block_loops, false);
            }
            if (bloop == 0) {
                first_pass_out = inpL;
            }
            if (bloop + 1 < block_loops) {
                float b_alpha = get_recurrent_block_alpha(bloop, block_loops, model.arch);
                ggml_tensor * s_orig = ggml_scale(ctx0, block_inp_orig, 1.0f - b_alpha);
                ggml_tensor * s_cur  = ggml_scale(ctx0, inpL, b_alpha);
                inpL = ggml_add(ctx0, s_orig, s_cur);
                if (momentum > 0.0f) {
                    ggml_tensor * delta_m = ggml_sub(ctx0, s_cur, s_orig);
                    inpL = ggml_add(ctx0, inpL, ggml_scale(ctx0, delta_m, momentum));
                }
                cb(inpL, "recurrent_block_inp", bloop);
            }
        }

        if (dual_stream && first_pass_out != nullptr) {
            ggml_tensor * prim_final = inpL;
            ggml_tensor * delta = ggml_sub(ctx0, prim_final, block_inp_orig);
            ggml_tensor * scaled_delta = ggml_scale(ctx0, delta, beta);
            inpL = ggml_sub(ctx0, block_inp_orig, scaled_delta);

            for (int il = block_start; il <= block_end; ++il) {
                build_layer(il, block_loops, block_loops, true);
            }
            alt_stream_out = inpL;

            ggml_tensor * consensus = ggml_add(ctx0,
                ggml_scale(ctx0, first_pass_out, 1.0f - gamma),
                ggml_scale(ctx0, alt_stream_out, gamma));
            cb(consensus, "recurrent_consensus", 0);

            inpL = ggml_add(ctx0,
                ggml_scale(ctx0, first_pass_out, 1.0f - exit_alpha),
                ggml_scale(ctx0, consensus, exit_alpha));
            cb(inpL, "recurrent_exit", 0);
        } else if (exit_alpha < 1.0f && first_pass_out != nullptr) {
            inpL = ggml_add(ctx0,
                ggml_scale(ctx0, first_pass_out, 1.0f - exit_alpha),
                ggml_scale(ctx0, inpL, exit_alpha));
            cb(inpL, "recurrent_exit", 0);
        }
    } else {
        for (int il = block_start; il <= block_end; ++il) {
            build_layer(il, 0, 1, false);
        }
    }

    for (int il = block_end + 1; il < n_layer; ++il) {
        build_layer(il, 0, 1, false);
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
