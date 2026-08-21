#include "models.h"

void llama_model_gemma2::load_arch_hparams(llama_model_loader & ml) {
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.n_swa = 4096; // default value of gemma 2
    uint32_t swa_period = 2;
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false);
    hparams.set_swa_pattern(swa_period);
    hparams.attn_soft_cap = true;
    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,          hparams.rope_freq_base_train_swa, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTN_LOGIT_SOFTCAPPING,      hparams.f_attn_logit_softcapping, false);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

    switch (hparams.n_layer()) {
        case 26: type = LLM_TYPE_2B; break;
        case 42: type = LLM_TYPE_9B; break;
        case 46: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
   }

    // ref: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/config.py#L173
    hparams.f_attention_scale = type == LLM_TYPE_27B
        ? 1.0f / std::sqrt(float(hparams.n_embd / hparams.n_head(0)))
        : 1.0f / std::sqrt(float(hparams.n_embd_head_k()));
}

void llama_model_gemma2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED); // same as tok_embd, duplicated to allow offloading

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gemma2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_gemma2::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_k();

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
    cb(inpL, "inp_scaled", -1);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int block_loops = get_recurrent_block_loops();
    auto [block_start, block_end] = get_recurrent_block_range(n_layer, model.arch);

    auto build_layer = [&](int il, int bloop = 0, int bloops = 1, bool is_alt = false) {
        ggml_tensor * inpSA = inpL;
        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

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
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_scale(ctx0, Qcur, hparams.f_attention_scale);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL, model.layers[il].wo_s,
                    Qcur, get_store_kv(bloop, bloops, is_alt) ? Kcur : nullptr, get_store_kv(bloop, bloops, is_alt) ? Vcur : nullptr, nullptr, nullptr, nullptr, 1.0f, il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        cur = build_norm(cur,
                model.layers[il].attn_post_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
        cb(sa_out, "sa_out", il);

        cur = build_norm(sa_out,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // feed-forward network
        {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }
        cur = build_norm(cur,
                model.layers[il].ffn_post_norm, NULL,
                LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", -1);

        cur = ggml_add(ctx0, cur, sa_out);

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

        for (int bloop = 0; bloop < block_loops; ++bloop) {
            for (int il = block_start; il <= block_end; ++il) {
                build_layer(il, bloop, block_loops, false);
            }
            if (bloop == 0) {
                first_pass_out = inpL;
            }
            if (bloop + 1 < block_loops) {
                ggml_tensor * mixed = ggml_add(ctx0,
                    ggml_scale(ctx0, block_inp_orig, 1.0f - alpha),
                    ggml_scale(ctx0, inpL, alpha));
                cb(mixed, "recurrent_block_inp", bloop);
                inpL = mixed;
            }
        }

        if (dual_stream && first_pass_out != nullptr) {
            ggml_tensor * delta = ggml_sub(ctx0, first_pass_out, block_inp_orig);
            ggml_tensor * scaled_delta = ggml_scale(ctx0, delta, beta);
            inpL = ggml_sub(ctx0, block_inp_orig, scaled_delta);

            for (int il = block_start; il <= block_end; ++il) {
                build_layer(il, block_loops, block_loops, true);
            }
            alt_stream_out = inpL;

            ggml_tensor * consensus = ggml_add(ctx0,
                ggml_scale(ctx0, first_pass_out, 1.0f - beta),
                ggml_scale(ctx0, alt_stream_out, beta));
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

    // final logit soft-capping
    cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
    cur = ggml_tanh(ctx0, cur);
    cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
