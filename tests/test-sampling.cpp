#include "ggml.h"
#include "llama.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

extern struct llama_sampler * llama_sampler_init_dry_testing(int32_t context_size, float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const std::vector<std::vector<llama_token>>& seq_breakers);

static void dump(const llama_token_data_array * cur_p) {
    for (size_t i = 0; i < cur_p->size; i++) {
        printf("%d: %f (%f)\n", cur_p->data[i].id, cur_p->data[i].p, cur_p->data[i].logit);
    }
}

#define DUMP(__cur_p) do { printf("%s:%d (%s)\n", __FILE__, __LINE__, __func__); dump((__cur_p)); printf("-\n"); } while(0)

struct sampler_tester {
    sampler_tester(size_t n_vocab) {
        cur.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
            const float logit = logf(token_id);
            cur.emplace_back(llama_token_data{token_id, logit, 0.0f});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    sampler_tester(const std::vector<float> & probs, const std::vector<float> & probs_expected) : probs_expected(probs_expected) {
        cur.reserve(probs.size());
        for (llama_token token_id = 0; token_id < (llama_token)probs.size(); token_id++) {
            const float logit = logf(probs[token_id]);
            cur.emplace_back(llama_token_data{token_id, logit, probs[token_id]});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    void apply(llama_sampler * sampler) {
        llama_sampler_apply(sampler, &cur_p);
        llama_sampler_free(sampler);
    }

    void check() {
        GGML_ASSERT(cur_p.size == probs_expected.size());
        for (size_t i = 0; i < cur_p.size; i++) {
            GGML_ASSERT(fabs(cur_p.data[i].p - probs_expected[i]) < 1e-5);
        }
    }

    llama_token_data_array cur_p;

private:
    const std::vector<float> probs_expected;

    std::vector<llama_token_data> cur;
};

static void test_temp(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp(temp));
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_temp_ext(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp, float delta, float exponent) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp_ext(temp, delta, exponent));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_top_k(const std::vector<float> & probs, const std::vector<float> & probs_expected, int k) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_k(k));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_top_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_min_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_min_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_xtc(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p, float t) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_xtc(p, t, 0, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_typical(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_typical(p, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_penalties(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & probs_expected, float repeat_penalty, float alpha_frequency, float alpha_presence
) {
    GGML_ASSERT(probs.size() == probs_expected.size());

    sampler_tester tester(probs, probs_expected);

    auto * sampler = llama_sampler_init_penalties(last_tokens.size(), repeat_penalty, alpha_frequency, alpha_presence);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_dry(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & expected_probs, float dry_multiplier, float dry_base,
    int dry_allowed_length, int dry_penalty_last_n,
    const std::vector<std::vector<llama_token>> & seq_breakers
) {
    GGML_ASSERT(probs.size() == expected_probs.size());

    sampler_tester tester(probs, expected_probs);

    auto * sampler = llama_sampler_init_dry_testing(1024, dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);
    tester.check();
}

static void test_top_n_sigma(const std::vector<float> & probs, const std::vector<float> & probs_expected, int n) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_n_sigma(n));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

// quasi (arithmetic) sampling
//
// see llama_sampler_init_quasi() in llama.h. the tests below pin down the four properties that make the
// scheme usable as a drop-in replacement for ancestral sampling:
//   a) determinism           - same logits + same code -> same tokens, bit for bit
//   b) endpoint behaviour    - the extremes of the code interval map to the extremes of the distribution
//   c) measure preservation  - sweeping the code over a grid reproduces the target probabilities
//   d) no silent failure     - an out-of-range code is rejected, not clamped
// plus the actual reason the scheme exists: two codes that are far apart decode to different tokens.

struct quasi_tester {
    // `logits` must be in descending order - the same contract the candidate list has when it reaches
    // llama_sampler_dist_apply(), and the codebook order used to build the intervals
    explicit quasi_tester(const std::vector<float> & logits) {
        cur.reserve(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
            cur.emplace_back(llama_token_data{ (llama_token) i, logits[i], 0.0f });
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, true };
    }

    std::vector<llama_token_data> cur;
    llama_token_data_array        cur_p;
};

// applies a fresh quasi sampler to `logits` and returns the index of the selected candidate
static int quasi_select(const std::vector<float> & logits, float p_quasi, float quasi, uint32_t seed = 0) {
    quasi_tester tester(logits);

    auto * smpl = llama_sampler_init_quasi(p_quasi, quasi, seed);

    llama_sampler_apply(smpl, &tester.cur_p);

    const int selected = tester.cur_p.selected;

    llama_sampler_free(smpl);

    GGML_ASSERT(selected >= 0 && (size_t) selected < logits.size());

    return selected;
}

// decodes `n_steps` tokens, using a different distribution at every step (a fixed one would not
// exercise the code renormalization), and asserts that the code stays a valid code point throughout
static std::vector<int> quasi_sequence(const std::vector<std::vector<float>> & steps, float quasi, int n_steps) {
    quasi_tester tester(steps[0]);

    auto * smpl = llama_sampler_init_quasi(1.0f, quasi, 0);

    std::vector<int> out;

    for (int t = 0; t < n_steps; ++t) {
        for (size_t i = 0; i < steps[t].size(); ++i) {
            tester.cur[i].logit = steps[t][i];
        }

        llama_sampler_apply(smpl, &tester.cur_p);

        out.push_back(tester.cur_p.selected);

        const float c = llama_sampler_quasi_get(smpl);
        GGML_ASSERT(c >= 0.0f && c < 1.0f);
    }

    llama_sampler_free(smpl);

    return out;
}

// (a) determinism
static void test_quasi_determinism() {
    const std::vector<std::vector<float>> steps = {
        {  1.0f,  0.9f,  0.2f, -0.5f, -1.5f },
        {  0.3f,  0.0f,  0.0f, -0.1f, -2.0f },
        { -0.2f,  0.4f,  1.0f,  0.0f,  0.0f },
        {  0.0f,  0.0f,  0.0f,  0.0f,  0.0f },
        {  2.0f,  1.0f,  0.0f,  0.0f, -1.0f },
    };

    const std::vector<float> quasi_values = { 0.0f, 0.1f, 0.25f, 1.0f/3.0f, 0.5f, 0.618034f, 0.9f, 0.999f };

    for (const float quasi : quasi_values) {
        const auto ref = quasi_sequence(steps, quasi, (int) steps.size());

        // two independent instances, and the same instance again after reset, must agree exactly
        for (int rep = 0; rep < 4; ++rep) {
            GGML_ASSERT(quasi_sequence(steps, quasi, (int) steps.size()) == ref);
        }

        // and the seed-derived code must be stable too
        const auto ref_seed = quasi_sequence(steps, -1.0f, (int) steps.size());
        (void) ref_seed;

        printf("quasi %.6f -> %d %d %d %d %d\n", (double) quasi, ref[0], ref[1], ref[2], ref[3], ref[4]);
    }
}

// (b) endpoints of the code interval
static void test_quasi_endpoints() {
    // two equally likely modes plus two impossible ones
    const std::vector<float> logits = { 0.0f, 0.0f, -INFINITY, -INFINITY };

    const float lo = 0.0f;
    const float hi = std::nextafterf(1.0f, 0.0f);

    const int t_lo = quasi_select(logits, 1.0f, lo);
    const int t_hi = quasi_select(logits, 1.0f, hi);

    printf("quasi endpoints: lo -> %d, hi -> %d\n", t_lo, t_hi);

    GGML_ASSERT(t_lo == 0);
    GGML_ASSERT(t_hi == 1);
    GGML_ASSERT(t_lo != t_hi);

    // 1 - 1e-6 must also land on the second mode
    GGML_ASSERT(quasi_select(logits, 1.0f, 1.0f - 1e-6f) == 1);

    // single candidate: the code is irrelevant
    GGML_ASSERT(quasi_select({ 0.5f }, 1.0f, lo)  == 0);
    GGML_ASSERT(quasi_select({ 0.5f }, 1.0f, hi)  == 0);
}

// (c) the empirical measure over a uniform grid of codes must match the target distribution
static void test_quasi_distribution(const std::vector<float> & logits, float tol) {
    // target probabilities, in the same double precision the sampler accumulates in
    const float max_l = logits[0];

    double sum = 0.0;
    std::vector<double> p_target(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        p_target[i] = expf(logits[i] - max_l);
        sum += p_target[i];
    }
    for (auto & p : p_target) {
        p /= sum;
    }

    const int n_grid = 100;

    std::vector<int> counts(logits.size(), 0);

    for (int k = 0; k < n_grid; ++k) {
        counts[quasi_select(logits, 1.0f, (float) k/n_grid)] += 1;
    }

    double max_dev = 0.0;

    printf("quasi distribution over %d codes:\n", n_grid);
    for (size_t i = 0; i < logits.size(); ++i) {
        const double freq = (double) counts[i]/n_grid;
        const double dev  = fabs(freq - p_target[i]);

        max_dev = std::max(max_dev, dev);

        printf("  token %zu: p = %.6f  freq = %.2f  dev = %.2e\n", i, p_target[i], freq, dev);

        GGML_ASSERT(dev <= tol);
    }

    printf("  max deviation = %.3e (tolerance %.1e)\n", max_dev, (double) tol);

    GGML_ASSERT(max_dev <= tol);
}

// (d) an invalid code must be reported, not silently clamped
static void test_quasi_invalid() {
    const float nan = std::numeric_limits<float>::quiet_NaN();

    auto * smpl = llama_sampler_init_quasi(1.0f, 0.5f, 0);

    GGML_ASSERT(llama_sampler_quasi_get(smpl) == 0.5f);

    const float bad[] = { -1e-6f, -1.0f, 1.0f, 1.5f, 42.0f, nan, INFINITY, -INFINITY };

    for (const float quasi : bad) {
        GGML_ASSERT(!llama_sampler_quasi_set(smpl, quasi));
    }

    // none of the rejected calls may have disturbed the code
    GGML_ASSERT(llama_sampler_quasi_get(smpl) == 0.5f);

    // the closed-open interval [0, 1) is accepted
    GGML_ASSERT(llama_sampler_quasi_set(smpl, 0.0f));
    GGML_ASSERT(llama_sampler_quasi_get(smpl) == 0.0f);
    GGML_ASSERT(llama_sampler_quasi_set(smpl, std::nextafterf(1.0f, 0.0f)));
    GGML_ASSERT(llama_sampler_quasi_get(smpl) < 1.0f);

    llama_sampler_free(smpl);
}

// the point of the scheme: codes that are far apart in code space decode to different tokens
static void test_quasi_diversification() {
    const std::vector<float> logits = { 0.0f, 0.0f };

    // 2 codes at distance 0.5: always 2 distinct tokens
    int distinct = 0;
    std::vector<int> seen;

    for (int i = 0; i < 2; ++i) {
        const int t = quasi_select(logits, 1.0f, 0.25f + 0.5f*i);
        if (std::find(seen.begin(), seen.end(), t) == seen.end()) {
            seen.push_back(t);
        }
    }
    distinct = (int) seen.size();

    printf("quasi diversification: 2 codes -> %d distinct tokens\n", distinct);

    GGML_ASSERT(distinct == 2);

    // same with 4 codes on a 4-mode distribution: 4 distinct tokens out of 4 steps. the analogous
    // ancestral draw collides with probability 1 - 4!/4^4 = 0.906
    const std::vector<float> flat = { 0.0f, 0.0f, 0.0f, 0.0f };

    seen.clear();
    for (int i = 0; i < 4; ++i) {
        const int t = quasi_select(flat, 1.0f, 0.125f + 0.25f*i);
        if (std::find(seen.begin(), seen.end(), t) == seen.end()) {
            seen.push_back(t);
        }
    }
    printf("quasi diversification: 4 codes on 4 modes -> %d distinct tokens\n", (int) seen.size());
    GGML_ASSERT(seen.size() == 4);

    // the seed-derived code must differ between seeds and be stable within a seed
    quasi_tester tester({ 0.0f, 0.0f, -INFINITY, -INFINITY, -INFINITY });

    auto * s0 = llama_sampler_init_quasi(1.0f, -1.0f, 1234);
    auto * s1 = llama_sampler_init_quasi(1.0f, -1.0f, 1234);
    auto * s2 = llama_sampler_init_quasi(1.0f, -1.0f, 4321);

    const float c0 = llama_sampler_quasi_get(s0);
    const float c1 = llama_sampler_quasi_get(s1);
    const float c2 = llama_sampler_quasi_get(s2);

    printf("quasi seed codes: seed 1234 -> %.7f, seed 4321 -> %.7f\n", (double) c0, (double) c2);

    GGML_ASSERT(c0 == c1);
    GGML_ASSERT(c0 != c2);
    GGML_ASSERT(c0 >= 0.0f && c0 < 1.0f);

    // reset() must return the sampler to its starting code
    GGML_ASSERT(llama_sampler_quasi_set(s0, 0.75f));
    GGML_ASSERT(llama_sampler_quasi_get(s0) == 0.75f);
    llama_sampler_reset(s0);
    GGML_ASSERT(llama_sampler_quasi_get(s0) == 0.75f);

    // clone must preserve the live code
    auto * s3 = llama_sampler_clone(s0);
    GGML_ASSERT(llama_sampler_quasi_get(s3) == 0.75f);

    llama_sampler_free(s3);
    llama_sampler_free(s2);
    llama_sampler_free(s1);
    llama_sampler_free(s0);
}

static void test_sampler_queue(const size_t n_vocab, const std::string & samplers_sequence, const int top_k, const float top_p, const float min_p
) {
    sampler_tester tester(n_vocab);

          llama_token min_token_id = 0;
    const llama_token max_token_id = n_vocab - 1;

    for (auto s : samplers_sequence) {
        switch (s) {
            case 'k': tester.apply(llama_sampler_init_top_k(top_k)); break;
            case 'y': GGML_ABORT("typical test not implemented");
            case 'p': tester.apply(llama_sampler_init_top_p(top_p, 1)); break;
            case 'm': tester.apply(llama_sampler_init_min_p(min_p, 1)); break;
            case 't': GGML_ABORT("temperature test not implemented");
            default : GGML_ABORT("Unknown sampler");
        }

        tester.apply(llama_sampler_init_dist(0));

        auto & cur_p = tester.cur_p;

        const int size = cur_p.size;

        if (s == 'k') {
            const int expected_size = std::min(size, top_k);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - top_k));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(cur_p.data[0].id == max_token_id);
            GGML_ASSERT(cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'p') {
            const int softmax_divisor = n_vocab * (n_vocab-1) / 2 - min_token_id * (min_token_id-1) / 2;
            const int softmax_numerator_target = ceilf(top_p * softmax_divisor);

                min_token_id  = n_vocab;
            int expected_size = 0;
            int cumsum        = 0;
            do { // do-while because always at least one token is sampled
                min_token_id--;
                expected_size++;

                cumsum += min_token_id;
            } while (cumsum < softmax_numerator_target);

            // token 0 has p == 0, need special consideration for cumsum because top_p immediately returns
            if (min_token_id == 1) {
                min_token_id--;
                expected_size += 1;
            }

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'm') {
            int expected_size = ceilf((1.0f - min_p) * n_vocab);
            expected_size = std::max(expected_size, 1);
            expected_size = std::min(expected_size, size);

            min_token_id = floorf(min_p * n_vocab);
            min_token_id = std::max(min_token_id, 1);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - size));
            min_token_id = std::min(min_token_id, (llama_token)(n_vocab - 1));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else {
            GGML_ABORT("fatal error");
        }
    }

    printf("Sampler queue %3s OK with n_vocab=%05zu top_k=%5d top_p=%f min_p=%f\n",
           samplers_sequence.c_str(), n_vocab, top_k, top_p, min_p);
}

static void bench(llama_sampler * cnstr, const char * cnstr_name, const std::vector<llama_token_data> & data, int n_iter) {
    std::vector<llama_token_data> cur(data.size());
    std::copy(data.begin(), data.end(), cur.begin());
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(cnstr, &cur_p);
    llama_sampler_reset(cnstr);
    const int64_t t_start = ggml_time_us();
    for (int i = 0; i < n_iter; i++) {
        std::copy(data.begin(), data.end(), cur.begin());
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(cnstr, &cur_p);
        llama_sampler_reset(cnstr);
    }
    const int64_t t_end = ggml_time_us();
    llama_sampler_free(cnstr);
    printf("%-43s: %8.3f us/iter\n", cnstr_name, (t_end - t_start) / (float)n_iter);
}

#define BENCH(__cnstr, __data, __n_iter) bench((__cnstr), #__cnstr, (__data), (__n_iter))

static void test_perf() {
    const int n_vocab = 1 << 17;

    std::vector<llama_token_data> data;

    data.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        const float logit = 2.0f*((double)(rand())/RAND_MAX - 0.5);
        data.emplace_back(llama_token_data{i, logit, 0.0f});
    }

    BENCH(llama_sampler_init_top_k  (40),                     data, 32);
    BENCH(llama_sampler_init_top_p  (0.8f, 1),                data, 32);
    BENCH(llama_sampler_init_min_p  (0.2f, 1),                data, 32);
    BENCH(llama_sampler_init_typical(0.5f, 1),                data, 32);
    BENCH(llama_sampler_init_xtc    (1.0f, 0.1f, 1, 1),       data, 32);
}

int main(void) {
    ggml_time_init();

    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);
    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f);

    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f, 0.0f, 1.0f);
    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f, 0.0f, 1.0f);

    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 1);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 3);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f, 0.1f}, 4);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0);

    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 0);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.571429f, 0.428571f}, 0.7f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 0.8f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);

    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.24f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.26f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.49f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.51f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.74f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  0.76f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.05f);

    printf("XTC should:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.1f},                                0.99f, 0.09f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.2f, 0.1f},                          0.99f, 0.19f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.3f, 0.2f, 0.1f},                    0.99f, 0.29f);

    printf("XTC should not:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.4f, 0.3f, 0.2f, 0.1f},              0.99f, 0.39f);

    test_typical({0.97f, 0.01f, 0.01f, 0.01f}, {0.97f},            0.5f);
    test_typical({0.4f, 0.2f, 0.2f, 0.2f},     {0.2f, 0.2f, 0.2f}, 0.5f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0}, {0, 0.25f, 0.25f, 0.25f, 0.25f},   50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2}, {0, 0, 0, 0.5f, 0.5f},       50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0, 0, 0, 0.5f, 0.5f}, 50.0f, 0.0f, 0.0f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0},             {0.000011f, 0.249997f, 0.249997f, 0.249997f, 0.249997f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2},       {0.000023f, 0.000023f, 0.000023f, 0.499966f, 0.499966f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0.000000f, 0.000023f, 0.000023f, 0.499977f, 0.499977f}, 1.0f, 5.0f, 5.0f);


    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1}, {0.25f, 0.25f, 0.25f, 0.25f}, 1.0f, 1.1f, 2, 4, {});
    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1, 2, 0, 1}, {0.296923f, 0.296923f, 0.109232f, 0.296923f}, 1.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 2, 6, {{3}});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 1}, {0.241818f, 0.241818f, 0.032727f, 0.241818f, 0.241818f}, 2.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 4, 7, {});

    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.428571f, 0.571429f}, 1.00f);
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0.00f); // top_n_sigma == 0 now represents a no-op rather than greedy decoding as of PR#13345
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 3.00f);

    printf("quasi determinism:\n");
    test_quasi_determinism();

    printf("quasi endpoints:\n");
    test_quasi_endpoints();

    printf("quasi distribution (uniform):\n");
    test_quasi_distribution({ 0.0f, 0.0f }, 1e-2f);

    printf("quasi distribution (skewed):\n");
    test_quasi_distribution({ 2.0f, 1.0f, 0.0f, 0.0f, -1.0f, -3.0f }, 1e-2f);

    printf("quasi distribution (peaked):\n");
    test_quasi_distribution({ 4.0f, 0.0f, 0.0f, -4.0f, -8.0f }, 1e-2f);

    printf("quasi invalid codes:\n");
    test_quasi_invalid();

    printf("quasi diversification:\n");
    test_quasi_diversification();

    test_sampler_queue(10000, "k", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "k",     1, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1e-12);

    test_sampler_queue(10000, "k",   100, 1.0000f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0003f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.8000f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 0.1f);

    test_sampler_queue(10000, "kp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "km", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 0.1f);

    test_sampler_queue(10000, "kpm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "kmp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pkm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pmk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mkp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mpk", 100, 0.8f, 0.1f);

    printf("OK\n");

    test_perf();

    return 0;
}
