// SPDX-License-Identifier: Apache-2.0 AND MIT
// Portions adapted from mlx-lm/models/gated_delta.py (mlx-lm#1559).
// Copyright © 2025 Apple Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Template params: T (input/output dtype), StT (state dtype), Dk, Dv, Hk, Hv
// Inputs: q, k, v, g, beta, state_in, cu_seqlens, slot_mapping, num_requests
// Outputs: y, state_out
// Grid: (32, Dv / 8, num_requests * Hv), Threadgroup: (32, 2, 1)
//
// Packed specialization of gdn_recurrent_prefill.metal for Dk == 128 and
// Dv % 8 == 0, adapting mlx-lm's packed recurrence layout (mlx-lm#1559)
// to ragged sequences and paged state. The generic kernel
// assigns one 32-lane SIMD-group to each value row, which leaves every lane
// with only Dk/32 state elements and pays two full-SIMD reductions per row
// and token.  This kernel instead packs eight independent value rows into a
// SIMD-group: four lanes own each row and each lane keeps Dk/4 contiguous
// state elements in registers, so a token costs two 4-lane butterflies
// instead of two 32-lane reductions.
//
// Preserve the generic kernel's four-element partial sums, then reduce
// with an ascending XOR butterfly: levels 1/2/4 are local, and levels 8/16
// become shuffle_xor(1/2) across the four lanes of each row. Whether this
// matches the generic simd_sum bit-for-bit depends on its compiler lowering.

constexpr int lanes_per_row = 4;
constexpr int rows_per_simdgroup = 32 / lanes_per_row;
constexpr int values_per_lane = Dk / lanes_per_row;
constexpr int partials_per_lane = values_per_lane / 4;
// The reduction tree below is written out for exactly 8 partials per lane;
// the Python launch gate restricts this kernel to Dk == 128.
static_assert(Dk == 128, "packed prefill kernel requires Dk == 128");
static_assert(Dv % rows_per_simdgroup == 0, "packed prefill requires Dv % 8 == 0");

auto n = thread_position_in_grid.z;
auto req_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);

auto lane = thread_index_in_simdgroup;
auto row_in_simdgroup = lane / lanes_per_row;
auto lane_in_row = lane & (lanes_per_row - 1);
auto row_group = thread_position_in_grid.y;
auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

if (req_idx >= (uint)num_requests) return;

uint slot_idx = (uint)slot_mapping[req_idx];
int64_t state_base = int64_t(slot_idx) * state_in_strides[0]
    + (hv_idx * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
int state_out_base =
    ((req_idx * Hv + hv_idx) * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

int seq_start = cu_seqlens[req_idx];
int seq_end = cu_seqlens[req_idx + 1];
int seq_len = seq_end - seq_start;

float state[values_per_lane];
for (int i = 0; i < values_per_lane; ++i) {
    state[i] = static_cast<float>(state_in[state_base + i]);
}

auto q_ = q + (seq_start * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
auto k_ = k + (seq_start * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
auto v_ = v + (seq_start * Hv + hv_idx) * Dv;
auto y_ = y + (seq_start * Hv + hv_idx) * Dv;
auto g_ = g + seq_start * Hv;
auto beta_ = beta + seq_start * Hv;

for (int t = 0; t < seq_len; ++t) {
    float g_val = static_cast<float>(g_[hv_idx]);

    // Each 4-element chain is one original lane's sequential accumulation.
    float part[partials_per_lane];
    for (int pb = 0; pb < partials_per_lane; ++pb) {
        float acc = 0.0f;
        for (int i = 0; i < 4; ++i) {
            int e = pb * 4 + i;
            state[e] = state[e] * g_val;
            acc += state[e] * static_cast<float>(k_[e]);
        }
        part[pb] = acc;
    }
    // Butterfly levels xor 1/2/4 stay inside this lane; levels xor 8/16
    // become the two row-group shuffles.
    float kv_mem =
        ((part[0] + part[1]) + (part[2] + part[3])) +
        ((part[4] + part[5]) + (part[6] + part[7]));
    kv_mem += simd_shuffle_xor(kv_mem, 1);
    kv_mem += simd_shuffle_xor(kv_mem, 2);

    float delta = (static_cast<float>(v_[dv_idx]) - kv_mem)
                  * static_cast<float>(beta_[hv_idx]);

    for (int pb = 0; pb < partials_per_lane; ++pb) {
        float acc = 0.0f;
        for (int i = 0; i < 4; ++i) {
            int e = pb * 4 + i;
            state[e] = state[e] + static_cast<float>(k_[e]) * delta;
            acc += state[e] * static_cast<float>(q_[e]);
        }
        part[pb] = acc;
    }
    float out =
        ((part[0] + part[1]) + (part[2] + part[3])) +
        ((part[4] + part[5]) + (part[6] + part[7]));
    out += simd_shuffle_xor(out, 1);
    out += simd_shuffle_xor(out, 2);
    if (lane_in_row == 0) {
        y_[dv_idx] = static_cast<T>(out);
    }

    q_ += Hk * Dk;
    k_ += Hk * Dk;
    v_ += Hv * Dv;
    y_ += Hv * Dv;
    g_ += Hv;
    beta_ += Hv;
}

for (int i = 0; i < values_per_lane; ++i) {
    state_out[state_out_base + i] = static_cast<StT>(state[i]);
}
