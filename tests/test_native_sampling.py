# SPDX-License-Identifier: Apache-2.0
"""Tests for MLX-native temperature/top-k/top-p sampling."""

import mlx.core as mx
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

from tests.stub_runner import make_stub_runner
from vllm_metal.v1 import sampling_batch
from vllm_metal.v1.native_constraints import apply_native_constraints
from vllm_metal.v1.native_penalties import apply_native_penalties
from vllm_metal.v1.sampling import Sampler
from vllm_metal.v1.sampling_batch import (
    SamplingBatch,
    native_sampling_key,
    sample_from_logits,
)
from vllm_metal.v1.sampling_policy import (
    params_allow_native_greedy,
    params_allow_native_random,
)

VOCAB_SIZE = 512
BATCH_SIZE = 4


def _masked_in_vocab_order(
    logits: mx.array,
    top_k: list[int],
    top_p: list[float],
    min_p: list[float] | None = None,
) -> mx.array:
    """The native path's candidate mask, scattered back into vocab order.

    Calls the production masking directly so these tests exercise what
    sampling runs, and only rearranges its output for comparison with vLLM.
    """
    sorted_desc, sorted_idx = SamplingBatch._sorted_candidate_logits(
        logits, top_k, top_p, min_p if min_p is not None else [0.0] * len(top_k)
    )
    return mx.put_along_axis(
        mx.full(logits.shape, -mx.inf, dtype=logits.dtype),
        sorted_idx,
        sorted_desc,
        axis=-1,
    )


def _random_params(**overrides) -> SamplingParams:
    defaults = {"temperature": 0.7, "top_k": 20, "top_p": 0.95}
    defaults.update(overrides)
    return SamplingParams(**defaults)


class TestNativeRandomEligibility:
    def test_plain_temperature_top_k_top_p_batch_is_eligible(self) -> None:
        params = [
            _random_params(temperature=0.7),
            _random_params(temperature=1.0),
        ]

        assert params_allow_native_random(params)

    def test_empty_specific_token_list_is_eligible(self) -> None:
        assert params_allow_native_random([_random_params(logprob_token_ids=[])])

    @pytest.mark.parametrize(
        ("label", "params_list"),
        [
            ("greedy request", [_random_params(temperature=0.0)]),
            (
                "mixed greedy and random",
                [_random_params(), _random_params(temperature=0.0)],
            ),
            (
                "mixed top_k",
                [_random_params(top_k=20), _random_params(top_k=40)],
            ),
            (
                "mixed top_p",
                [_random_params(top_p=0.95), _random_params(top_p=0.8)],
            ),
            (
                "mixed min_p",
                [_random_params(min_p=0.1), _random_params(min_p=0.0)],
            ),
        ],
    )
    def test_per_row_masks_keep_mixed_batches_eligible(
        self, label, params_list
    ) -> None:
        """Masks are per row: one request must never push the whole batch
        onto the torch sampler (that serialises every decode step)."""
        assert params_allow_native_random(params_list)

    @pytest.mark.parametrize(
        ("label", "params_list"),
        [
            ("empty batch", []),
            ("seeded request", [_random_params(seed=7)]),
            ("frequency penalty", [_random_params(frequency_penalty=0.5)]),
            ("presence penalty", [_random_params(presence_penalty=0.5)]),
            ("repetition penalty", [_random_params(repetition_penalty=1.2)]),
            ("sample logprobs", [_random_params(logprobs=1)]),
            (
                "logprob token ids",
                [_random_params(logprob_token_ids=[1, 2])],
            ),
            (
                "allowed token ids",
                [_random_params(allowed_token_ids=[1, 2])],
            ),
            ("logit bias", [_random_params(logit_bias={1: 5.0})]),
            ("min tokens still pending", [_random_params(min_tokens=2)]),
        ],
    )
    def test_ineligible_batches_fall_back(self, label, params_list) -> None:
        assert not params_allow_native_random(params_list)

    def test_bad_words_fall_back(self) -> None:
        params = _random_params()
        params._bad_words_token_ids = [[99]]

        assert not params_allow_native_random([params])

    def test_min_tokens_returns_to_native_random_once_reached(self) -> None:
        params = [_random_params(min_tokens=2), _random_params()]
        assert not params_allow_native_random(params, [1, 0])
        assert params_allow_native_random(params, [2, 0])


class TestTopKTopPMaskParity:
    """The MLX mask must keep exactly the candidates vLLM's mask keeps."""

    @pytest.mark.parametrize(
        ("top_k", "top_p"),
        [
            (20, 1.0),
            (0, 0.95),
            (20, 0.95),
            (1, 1.0),
            (0, 1.0),
            (VOCAB_SIZE, 0.5),
            (0, 0.999999),
        ],
    )
    def test_mask_matches_vllm_reference(self, top_k: int, top_p: float) -> None:
        logits = mx.random.normal((BATCH_SIZE, VOCAB_SIZE), key=mx.random.key(42))
        mx.eval(logits)
        logits_torch = torch.tensor(logits.tolist())
        k_arg = (
            None
            if top_k <= 0 or top_k >= VOCAB_SIZE
            else torch.full((BATCH_SIZE,), top_k)
        )
        p_arg = None if top_p == 1.0 else torch.full((BATCH_SIZE,), top_p)

        masked_mlx = _masked_in_vocab_order(
            logits, [top_k] * BATCH_SIZE, [top_p] * BATCH_SIZE
        )
        mx.eval(masked_mlx)
        masked_ref = apply_top_k_top_p(logits_torch.clone(), k_arg, p_arg)

        kept_mlx = [[v != float("-inf") for v in row] for row in masked_mlx.tolist()]
        kept_ref = (masked_ref != float("-inf")).tolist()
        assert kept_mlx == kept_ref

    def test_top_p_boundary_ties_are_masked_positionally(self) -> None:
        """Boundary ties must not all survive (vLLM masks sorted positions).

        Four equal logits at top_p=0.25: each holds 0.25 mass, so exactly
        one sorted position has leading mass < 0.25 and only one survives.
        """
        logits = mx.array([[2.0, 2.0, 2.0, 2.0]])
        logits_torch = torch.full((1, 4), 2.0)

        masked = _masked_in_vocab_order(logits, [0], [0.25])
        mx.eval(masked)
        masked_ref = apply_top_k_top_p(logits_torch, None, torch.full((1,), 0.25))

        kept_count = sum(v != float("-inf") for v in masked.tolist()[0])
        kept_count_ref = int((masked_ref != float("-inf")).sum().item())
        assert kept_count == kept_count_ref == 1

    def test_per_row_masks_match_vllm_reference(self) -> None:
        """Each row keeps exactly the candidates vLLM keeps for its own k/p."""
        logits = mx.random.normal((4, VOCAB_SIZE), key=mx.random.key(9))
        mx.eval(logits)
        top_k = [20, 0, 7, VOCAB_SIZE]
        top_p = [1.0, 0.9, 0.5, 0.75]
        masked_mlx = _masked_in_vocab_order(logits, top_k, top_p)
        mx.eval(masked_mlx)
        logits_torch = torch.tensor(logits.tolist())
        k_arg = torch.tensor([k if 0 < k < VOCAB_SIZE else VOCAB_SIZE for k in top_k])
        masked_ref = apply_top_k_top_p(logits_torch.clone(), k_arg, torch.tensor(top_p))

        kept_mlx = [[v != float("-inf") for v in row] for row in masked_mlx.tolist()]
        kept_ref = (masked_ref != float("-inf")).tolist()
        assert kept_mlx == kept_ref

    def test_mixed_top_k_rows_keep_ties_at_their_own_kth_value(self) -> None:
        """Each row's ties at its own k-th largest survive, whatever the
        other rows in the batch ask for."""
        logits = mx.array(
            [[3.0, 2.0, 2.0, 2.0, 1.0, 0.0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]]
        )
        sorted_desc, sorted_idx = SamplingBatch._sorted_candidate_logits(
            logits, [2, 4], [1.0, 1.0], [0.0, 0.0]
        )
        mx.eval(sorted_desc, sorted_idx)
        kept_rows = [
            {
                int(t)
                for t, v in zip(idx_row, val_row, strict=True)
                if v != float("-inf")
            }
            for idx_row, val_row in zip(
                sorted_idx.tolist(), sorted_desc.tolist(), strict=True
            )
        ]
        assert kept_rows == [{0, 1, 2, 3}, {0, 1, 2, 3}]

    def test_single_row_top_k_keeps_ties_at_the_kth_value(self) -> None:
        """One row on its own must keep the same boundary ties vLLM keeps.

        Top-k masks by value, so every logit equal to the k-th largest
        survives: ``top_k=2`` over ``[3, 2, 2, 2, 1, 0]`` keeps four tokens.
        A candidate set capped at the row's own ``top_k`` would keep two.
        """
        logits = mx.array([[3.0, 2.0, 2.0, 2.0, 1.0, 0.0]])
        sorted_desc, sorted_idx = SamplingBatch._sorted_candidate_logits(
            logits, [2], [1.0], [0.0]
        )
        mx.eval(sorted_desc, sorted_idx)
        kept = {
            int(t)
            for t, v in zip(
                sorted_idx.tolist()[0], sorted_desc.tolist()[0], strict=True
            )
            if v != float("-inf")
        }

        masked_ref = apply_top_k_top_p(
            torch.tensor([[3.0, 2.0, 2.0, 2.0, 1.0, 0.0]]),
            torch.tensor([2]),
            None,
        )
        kept_ref = {
            i for i, v in enumerate(masked_ref.tolist()[0]) if v != float("-inf")
        }
        assert kept == kept_ref == {0, 1, 2, 3}


class TestMlxRandomTokens:
    def test_greedy_row_with_tied_maximum_stays_on_the_argmax(self) -> None:
        """A greedy row in a mixed batch picks the lowest token ID on an exact tie."""
        logits = mx.array([[1.0, 5.0, 5.0, 0.0], [1.0, 2.0, 3.0, 4.0]])
        params = [
            _random_params(temperature=0.0),
            _random_params(temperature=1.0, top_k=0, top_p=0.9),
        ]
        expected = int(mx.argmax(logits[0]).item())

        drawn = set()
        for seed in range(50):
            tokens = SamplingBatch._native_random_tokens(
                logits, params, mx.random.key(seed)
            )
            mx.eval(tokens)
            drawn.add(int(tokens.tolist()[0]))

        assert drawn == {expected}

    def test_mixed_batch_rows_follow_their_own_params(self) -> None:
        """A greedy row sticks to its argmax and a narrow top-k row stays inside
        its candidates while a wide row keeps exploring — all in one draw."""
        logits = mx.random.normal((3, VOCAB_SIZE), key=mx.random.key(21))
        mx.eval(logits)
        argmax = int(mx.argmax(logits[0]).item())
        narrow_candidates = {int(i) for i in mx.argsort(logits[1])[::-1][:4].tolist()}
        params = [
            _random_params(temperature=0.0),
            _random_params(temperature=1.5, top_k=4, top_p=1.0),
            _random_params(temperature=2.0, top_k=0, top_p=1.0),
        ]
        assert params_allow_native_random(params)

        greedy_row, narrow_row, wide_row = set(), set(), set()
        key = mx.random.key(1)
        for _ in range(200):
            key, subkey = mx.random.split(key)
            tokens = SamplingBatch._native_random_tokens(logits, params, subkey)
            mx.eval(tokens)
            g, n, w = tokens.tolist()
            greedy_row.add(g)
            narrow_row.add(n)
            wide_row.add(w)

        assert greedy_row == {argmax}
        assert narrow_row <= narrow_candidates and len(narrow_row) > 1
        assert len(wide_row) > len(narrow_candidates)

    def test_samples_stay_inside_candidate_set(self) -> None:
        candidate_count = 8
        logits = mx.full((BATCH_SIZE, VOCAB_SIZE), -100.0)
        logits[:, :candidate_count] = mx.linspace(5.0, 1.0, candidate_count)[None, :]
        mx.eval(logits)
        params = [_random_params(top_k=candidate_count)] * BATCH_SIZE

        sampled = []
        key = mx.random.key(0)
        for _ in range(200):
            key, subkey = mx.random.split(key)
            tokens = SamplingBatch._native_random_tokens(logits, params, subkey)
            mx.eval(tokens)
            sampled.extend(tokens.tolist())

        assert all(0 <= token < candidate_count for token in sampled)
        assert len(set(sampled)) > 1

    def test_same_key_is_deterministic(self) -> None:
        logits = mx.random.normal((BATCH_SIZE, VOCAB_SIZE), key=mx.random.key(3))
        mx.eval(logits)
        params = [_random_params()] * BATCH_SIZE
        key = mx.random.key(11)

        first = SamplingBatch._native_random_tokens(logits, params, key)
        second = SamplingBatch._native_random_tokens(logits, params, key)
        mx.eval(first, second)

        assert first.tolist() == second.tolist()

    def test_per_row_temperatures_shape_distinct_distributions(self) -> None:
        """Row temperatures are per-row facts: a near-zero-temperature row
        must stick to the argmax while a hot row keeps exploring."""
        logits = mx.array([[5.0, 4.5, 4.0, 3.5]] * 2)
        mx.eval(logits)
        params = [
            _random_params(temperature=0.01, top_k=0, top_p=1.0),
            _random_params(temperature=2.0, top_k=0, top_p=1.0),
        ]

        cold_row, hot_row = set(), set()
        key = mx.random.key(7)
        for _ in range(100):
            key, subkey = mx.random.split(key)
            tokens = SamplingBatch._native_random_tokens(logits, params, subkey)
            mx.eval(tokens)
            cold_row.add(tokens.tolist()[0])
            hot_row.add(tokens.tolist()[1])

        assert cold_row == {0}
        assert len(hot_row) > 1

    def test_distribution_tracks_probabilities(self) -> None:
        probs = [0.5, 0.3, 0.2]
        logits = mx.log(mx.array([probs] * 64))
        mx.eval(logits)
        params = [_random_params(temperature=1.0, top_k=0, top_p=1.0)] * 64

        counts = [0, 0, 0]
        key = mx.random.key(5)
        draws = 0
        for _ in range(400):
            key, subkey = mx.random.split(key)
            tokens = SamplingBatch._native_random_tokens(logits, params, subkey)
            mx.eval(tokens)
            for token in tokens.tolist():
                counts[token] += 1
                draws += 1

        for index, expected in enumerate(probs):
            assert abs(counts[index] / draws - expected) < 0.02


class TestNativeSamplingKey:
    @pytest.mark.parametrize(
        ("seed", "step", "legacy_key_seed"),
        [
            (0, 0, 0xCFD8C22A91380BA0),
            (7, 3, 0x3D1DD505FE98FC2C),
            (-7, 3, 0xA9DE8C3DA842F403),
        ],
    )
    def test_engine_keys_are_separate_from_legacy_request_keys(
        self, seed: int, step: int, legacy_key_seed: int
    ) -> None:
        """Preserve #1233's request keys without sharing the engine stream."""
        request_key = native_sampling_key(seed, step)
        engine_key = native_sampling_key(seed, step, domain="engine")
        legacy_key = mx.random.key(legacy_key_seed)
        mx.eval(request_key, engine_key, legacy_key)
        assert request_key.tolist() == legacy_key.tolist()
        assert engine_key.tolist() != request_key.tolist()

    @pytest.mark.parametrize("seed", [0, 42, -7])
    def test_engine_seed_and_step_reproduce_keys_and_tokens(self, seed: int) -> None:
        runner = make_stub_runner(
            _native_sampling_enabled=True, _native_sampling_step=11
        )
        runner.model_config.seed = seed
        key = runner._next_native_sample_key()
        same_key = native_sampling_key(seed, 11, domain="engine")
        next_key = runner._next_native_sample_key()
        logits = mx.zeros((8, VOCAB_SIZE))
        params = [_random_params(temperature=0.8, top_k=0, top_p=1.0)] * 8
        tokens = [
            SamplingBatch.native_sample_tokens(
                logits,
                params,
                vocab_size=VOCAB_SIZE,
                next_native_rng_key=lambda k=k: k,
                output_lengths=[0] * len(params),
            )
            for k in (key, same_key, next_key)
        ]
        mx.eval(key, same_key, next_key, *tokens)
        assert key.tolist() == same_key.tolist()
        assert key.tolist() != next_key.tolist()
        assert tokens[0].tolist() == tokens[1].tolist()
        assert tokens[0].tolist() != tokens[2].tolist()

    @pytest.mark.parametrize("seeds", [(0, 42), (0, 0), (42, 42)])
    def test_runner_keys_respect_engine_seed(self, seeds: tuple[int, int]) -> None:
        keys = []
        for seed in seeds:
            runner = make_stub_runner(
                _native_sampling_enabled=True, _native_sampling_step=11
            )
            runner.model_config.seed = seed
            keys.append(runner._next_native_sample_key())
        mx.eval(*keys)
        assert (keys[0].tolist() == keys[1].tolist()) == (seeds[0] == seeds[1])


class TestSeededNativeSampling:
    """Seeded requests draw from a seed-derived per-request MLX key."""

    @staticmethod
    def _sample(
        logits: mx.array,
        params: list[SamplingParams],
        output_lengths: list[int],
        engine_seed: int = 0,
    ) -> list[int]:
        key = mx.random.key(engine_seed)
        tokens = SamplingBatch.native_sample_tokens(
            logits,
            params,
            vocab_size=VOCAB_SIZE,
            next_native_rng_key=lambda: key,
            output_lengths=output_lengths,
            allow_seed=True,
        )
        mx.eval(tokens)
        return tokens.tolist()

    @pytest.mark.parametrize(
        ("params", "steps_per_call"),
        [
            ([_random_params(seed=1), _random_params(seed=2)], 0),
            ([_random_params(seed=1), _random_params(temperature=0.0)], 0),
            ([_random_params(seed=1), _random_params()], 1),
        ],
        ids=["all-seeded", "seeded-plus-greedy", "seeded-plus-unseeded"],
    )
    def test_engine_counter_advances_only_for_unseeded_random_rows(
        self, params: list[SamplingParams], steps_per_call: int
    ) -> None:
        logits = mx.random.normal((len(params), VOCAB_SIZE), key=mx.random.key(3))
        runner = make_stub_runner(_native_sampling_enabled=True)
        steps = 4
        for n in range(steps):
            tokens = SamplingBatch.native_sample_tokens(
                logits,
                params,
                vocab_size=VOCAB_SIZE,
                next_native_rng_key=runner._next_native_sample_key,
                output_lengths=[n] * len(params),
                allow_seed=True,
            )
            mx.eval(tokens)
        assert runner._native_sampling_step == steps_per_call * steps

    def test_seeded_rows_are_eligible_only_with_allow_seed(self) -> None:
        params = [_random_params(seed=7), _random_params()]
        assert not params_allow_native_random(params)
        assert params_allow_native_random(params, allow_seed=True)

    def test_row_key_is_pure_and_position_dependent(self) -> None:
        a = native_sampling_key(7, 3)
        b = native_sampling_key(7, 3)
        mx.eval(a, b)
        assert a.tolist() == b.tolist()
        others = [
            native_sampling_key(7, 4),
            native_sampling_key(8, 3),
            native_sampling_key(-7, 3),
        ]
        mx.eval(*others)
        assert all(o.tolist() != a.tolist() for o in others)

    def test_same_seed_reproduces_tokens(self) -> None:
        logits = mx.random.normal((1, VOCAB_SIZE), key=mx.random.key(5))
        params = [_random_params(temperature=1.5, top_k=0, top_p=1.0, seed=42)]
        runs = [
            [self._sample(logits, params, [n], engine_seed=e)[0] for n in range(32)]
            for e in (0, 1)
        ]
        assert runs[0] == runs[1]
        assert len(set(runs[0])) > 1

    def test_different_seeds_differ(self) -> None:
        logits = mx.random.normal((1, VOCAB_SIZE), key=mx.random.key(5))

        def stream(seed: int) -> list[int]:
            params = [_random_params(temperature=1.5, top_k=0, top_p=1.0, seed=seed)]
            return [self._sample(logits, params, [n])[0] for n in range(32)]

        assert stream(1) != stream(2)

    def test_seeded_row_independent_of_batch_neighbors(self) -> None:
        logits = mx.random.normal((4, VOCAB_SIZE), key=mx.random.key(9))
        mx.eval(logits)
        seeded = _random_params(temperature=1.2, top_k=50, top_p=0.9, seed=123)
        alone = self._sample(logits[2:3], [seeded], [5])[0]
        neighbors = [
            _random_params(temperature=0.0),
            _random_params(temperature=0.9, min_p=0.05),
            seeded,
            _random_params(temperature=1.0, seed=999),
        ]
        for engine_seed in range(4):
            mixed = self._sample(logits, neighbors, [0, 3, 5, 1], engine_seed)
            assert mixed[2] == alone
            assert mixed[0] == int(mx.argmax(logits[0]).item())

    def test_seeded_draws_respect_candidate_set(self) -> None:
        logits = mx.random.normal((1, VOCAB_SIZE), key=mx.random.key(13))
        mx.eval(logits)
        top = {int(i) for i in mx.argsort(logits[0])[::-1][:4].tolist()}
        params = [_random_params(temperature=2.0, top_k=4, top_p=1.0, seed=3)]
        drawn = {self._sample(logits, params, [n])[0] for n in range(64)}
        assert drawn <= top and len(drawn) > 1

    def test_unseeded_rows_unchanged_by_seeded_neighbor(self) -> None:
        """Unseeded rows keep the shared batched draw, bitwise."""
        logits = mx.random.normal((3, VOCAB_SIZE), key=mx.random.key(17))
        mx.eval(logits)
        plain = [_random_params(temperature=1.0, top_k=0, top_p=1.0)] * 3
        with_seed = [plain[0], _random_params(seed=5), plain[2]]
        key = mx.random.key(77)
        base = SamplingBatch._native_random_tokens(logits, plain, key)
        mixed = SamplingBatch.native_sample_tokens(
            logits,
            with_seed,
            vocab_size=VOCAB_SIZE,
            next_native_rng_key=lambda: key,
            output_lengths=[0, 0, 0],
            allow_seed=True,
        )
        mx.eval(base, mixed)
        assert [base.tolist()[i] for i in (0, 2)] == [mixed.tolist()[i] for i in (0, 2)]

    def test_native_call_without_allow_seed_rejects_seeded_rows(self) -> None:
        logits = mx.zeros((1, VOCAB_SIZE))
        with pytest.raises(RuntimeError, match="gate desync"):
            SamplingBatch.native_sample_tokens(
                logits,
                [_random_params(seed=1)],
                vocab_size=VOCAB_SIZE,
                next_native_rng_key=lambda: mx.random.key(0),
                output_lengths=[0],
            )


def _sampling_golden_cases():
    # The two sampled IDs per case come from the pinned vLLM FP32 Triton sampler:
    # https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/sample/gumbel.py#L165-L205
    for vocab_size, sampled_ids in (
        (3, (1, 1)),
        (1025, (524, 991)),
        (6562, (3619, 3169)),
    ):
        logits = ((torch.arange(vocab_size) % 17).float() / 8).repeat(5, 1)
        logits[0].zero_()
        logits[:, 2::5] = -float("inf")
        # An invalid row must still return a real ID, even across reordered tiles.
        logits[3:].fill_(float("nan"))
        logits[4, -1] = -20.0
        seeds, positions, temperatures = (
            [42, -1, (1 << 62) + 31, 42, 42],
            [0, 2, 17, 0, 0],
            [0.0, 0.8, 1.5, 1.0, 1.0],
        )
        expected = [0, *sampled_ids, 0, vocab_size - 1]
        yield logits, seeds, positions, temperatures, expected


def test_mlx_sampler_matches_upstream_with_padding_and_reordering():
    """Padded MLX rows preserve upstream draws and original token-ID ties."""
    from vllm_metal.metal import get_ops

    for logits, seeds, positions, temperatures, expected in _sampling_golden_cases():
        values = mx.array(logits.numpy())
        order = mx.argsort(values, axis=-1)[:, ::-1]
        values = mx.pad(
            mx.take_along_axis(values, order, axis=-1),
            ((0, 0), (1, 2)),
            constant_values=1000.0,
        )[:, 1:-2]
        mlx_ids = get_ops().gumbel_sample(
            values,
            mx.array(seeds, dtype=mx.int64),
            mx.array(positions, dtype=mx.int64),
            mx.array(temperatures, dtype=mx.float32),
            order,
        )
        assert mlx_ids.tolist() == expected, logits.shape[-1]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_mps_sampler_matches_upstream_with_padding(run_in_spawn_process):
    """Padded MPS rows preserve upstream draws and token-ID ties."""
    run_in_spawn_process(_check_mps_sampler_against_upstream)


def _check_mps_sampler_against_upstream():
    from vllm.v1.worker.gpu.sample import sampler

    from vllm_metal.pytorch_backend.bindings import install, register_kernel_overrides

    register_kernel_overrides()
    install()
    for logits, seeds, positions, temperatures, expected in _sampling_golden_cases():
        mps_logits = torch.nn.functional.pad(logits.to("mps"), (1, 2), value=1000.0)
        mps_ids = sampler.gumbel_sample(
            mps_logits[:, 1:-2],
            torch.arange(len(seeds), dtype=torch.int32, device="mps"),
            torch.tensor(temperatures, device="mps"),
            torch.tensor(seeds, device="mps"),
            torch.tensor(positions, device="mps"),
            True,
            False,
        )
        assert mps_ids.cpu().tolist() == expected, logits.shape[-1]


def test_broadcast_logits_use_per_row_candidate_ids():
    from vllm_metal.metal import get_ops

    # The same strided scores select different token IDs in the two rows.
    scores = mx.array([[-9.0, 1.0, -9.0, 5.0, -9.0, 2.0]])[:, 1::2]
    tokens = get_ops().gumbel_sample(
        mx.broadcast_to(scores, (2, 3)),
        mx.zeros(2, mx.int64),
        mx.zeros(2, mx.int64),
        mx.zeros(2, mx.float32),
        mx.array([[1, 0, 2], [0, 2, 1]], dtype=mx.uint32),
    )
    assert tokens.tolist() == [0, 2]


class TestNativeConstraints:
    @pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
    def test_mask_matches_vendored_sampler_for_mixed_rows(self, dtype):
        params = [
            SamplingParams(allowed_token_ids=[0, 1, 2, 2, 3]),
            SamplingParams(),
            SamplingParams(),
            SamplingParams(),
        ]
        for sp in params[:3]:
            sp._bad_words_token_ids = [[4], [1, 2], [2, 1, 3], [1, 2, 1, 0]]
        raw = [[0.0, 1.0, 2.0, 3.0, 4.0]] * 4
        outputs = [[1], [2, 1], [1, 2, 1], []]
        batch = SamplingBatch(params, [[1, 2, 1]] * 4, outputs, vocab_size=5)
        expected = Sampler().apply_logits_processors(
            torch.tensor(raw), batch.make_sampling_metadata(), False
        )
        logits = mx.array(raw, dtype=dtype)
        masked = apply_native_constraints(logits, params, outputs)

        assert masked is not None
        assert masked.dtype == dtype
        assert masked.tolist() == expected.tolist()
        # Pin the suffix result independently of the shared Torch matcher.
        assert mx.isinf(masked).tolist() == [
            [False, False, True, False, True],
            [False, False, True, True, True],
            [True, False, True, True, True],
            [False, False, False, False, False],
        ]
        assert logits.tolist() == raw

    @pytest.mark.parametrize("temperature", [0.0, 0.8])
    def test_sync_uses_current_output_suffix_and_real_vocab(
        self, monkeypatch, temperature
    ):
        sp = SamplingParams(
            temperature=temperature, allowed_token_ids=[1, 2], top_k=1, seed=7
        )
        sp._bad_words_token_ids = [[0, 2]]
        sampler = Sampler()
        monkeypatch.setattr(
            sampler, "forward", lambda *_: pytest.fail("unexpected Torch fallback")
        )
        for history, expected in [([], 2), ([0], 1), ([0, 1], 2)]:
            batch = SamplingBatch([sp], [[0]], [history], vocab_size=4)
            result = sample_from_logits(
                mx.array([[10.0, 1.0, 3.0, 4.0, 100.0]]),
                batch,
                sampler,
                next_native_rng_key=lambda: pytest.fail(
                    "seeded row used the engine key"
                ),
            )
            assert result.token_ids == [expected]
        assert not params_allow_native_greedy([sp], [1])
        assert not params_allow_native_random([sp], [1], allow_seed=True)

    @pytest.mark.parametrize(
        ("allowed", "bad_words", "history"),
        [
            ([1], [[1]], []),
            (None, [[0], [1]], []),
            ([0, 1], [[0, 1]], [-1]),
        ],
        ids=["allowlist-exhausted", "vocab-exhausted", "pending-history"],
    )
    def test_constraint_fallback_does_not_consume_native_rng(
        self, monkeypatch, allowed, bad_words, history
    ):
        params = SamplingParams(temperature=0.8, allowed_token_ids=allowed)
        params._bad_words_token_ids = bad_words
        batch = SamplingBatch([params], [[0]], [history], vocab_size=2)
        logits = mx.array([[2.0, 1.0]])
        sampler = Sampler()
        forward = sampler.forward
        seen_devices = []

        def record(logits_torch, metadata):
            seen_devices.append(logits_torch.device.type)
            return forward(logits_torch, metadata)

        monkeypatch.setattr(sampler, "forward", record)
        sample_from_logits(
            logits,
            batch,
            sampler,
            next_native_rng_key=lambda: pytest.fail("fallback consumed the engine key"),
        )
        assert seen_devices == ["cpu"]

    @pytest.mark.parametrize(
        ("allowed", "bad_words"), [([-1], []), ([2], []), (None, [[2]])]
    )
    def test_invalid_indices_stay_out_of_the_mlx_scatter(self, allowed, bad_words):
        params = SamplingParams(allowed_token_ids=allowed)
        params._bad_words_token_ids = bad_words
        assert apply_native_constraints(mx.array([[0.0, 1.0]]), [params], [[]]) is None


class TestNativePenalties:
    @pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
    def test_counts_signs_and_prompt_membership_match_torch(self, dtype):
        params = [
            SamplingParams(
                repetition_penalty=1.5, frequency_penalty=0.5, presence_penalty=-0.25
            ),
            SamplingParams(
                repetition_penalty=0.5, frequency_penalty=-0.5, presence_penalty=0.75
            ),
            SamplingParams(),
            SamplingParams(frequency_penalty=0.25),
        ]
        prompts = [[0, 0, 2], [1, 3], [], [0] * 20]
        outputs = [[1, 1, 1, 2], [2, 2, 4], [-1], [4] * 257]
        raw = [[4.0, 3.0, -2.0, 0.0, -1.0, float("-inf")]] * 4
        batch = SamplingBatch(params, prompts, outputs, vocab_size=6)
        expected = Sampler.apply_penalties(
            torch.tensor(raw), batch.make_sampling_metadata(), outputs
        )
        logits = mx.array(raw, dtype=dtype)
        actual = apply_native_penalties(logits, params, prompts, outputs)

        assert actual is not None
        assert actual.dtype == mx.float32
        torch.testing.assert_close(torch.tensor(actual.tolist()), expected)
        assert actual[0, 0].item() == pytest.approx(4 / 1.5)
        assert actual[0, 1].item() == pytest.approx(0.75)
        assert actual[3, 4].item() == pytest.approx(-65.25)
        assert logits.tolist() == raw

    @pytest.mark.parametrize("random_row", [False, True])
    def test_sync_penalties_keep_seed_and_mixed_row_alignment(
        self, monkeypatch, random_row
    ):
        params = [
            SamplingParams(temperature=0, repetition_penalty=2),
            SamplingParams(
                temperature=0.8 if random_row else 0,
                top_k=1,
                frequency_penalty=1,
                seed=17,
            ),
        ]
        batch = SamplingBatch(params, [[0], [0]], [[], [0, 0]], vocab_size=3)
        sampler = Sampler()
        monkeypatch.setattr(
            sampler, "forward", lambda *_: pytest.fail("unexpected Torch fallback")
        )
        logits = mx.array([[4.0, 3.0, 0.0, 99.0], [4.0, 3.0, 0.0, 99.0]])
        result = sample_from_logits(
            logits,
            batch,
            sampler,
            next_native_rng_key=lambda: pytest.fail("seeded row used the engine key"),
        )
        assert result.token_ids == [1, 1]
        assert not params_allow_native_greedy(params, batch.output_lengths)
        assert not params_allow_native_random(
            params, batch.output_lengths, allow_seed=True
        )

    @pytest.mark.parametrize("temperature", [0.0, 0.8])
    def test_sync_applies_masks_before_penalties(self, monkeypatch, temperature):
        sp = SamplingParams(
            temperature=temperature,
            top_k=1,
            seed=17,
            allowed_token_ids=[0, 1, 2, 3],
            repetition_penalty=2,
            frequency_penalty=0.5,
            presence_penalty=0.25,
        )
        sp._bad_words_token_ids = [[1, 0]]
        logits = mx.array([[10.0, -4.0, 6.0, 4.0, 9.0, 100.0]])
        batch = SamplingBatch([sp], [[2]], [[1]], vocab_size=5)
        sampler = Sampler()
        monkeypatch.setattr(
            sampler, "forward", lambda *_: pytest.fail("unexpected Torch fallback")
        )

        def check_masked_scores(scores, *args):
            assert scores.tolist() == [[-mx.inf, -4.0, 6.0, 4.0, -mx.inf]]
            return apply_native_penalties(scores, *args)

        monkeypatch.setattr(
            sampling_batch, "apply_native_penalties", check_masked_scores
        )
        result = sample_from_logits(
            logits, batch, sampler, next_native_rng_key=lambda: mx.random.key(0)
        )
        assert result.token_ids == [3]

    @pytest.mark.parametrize(
        "exhausted", [False, True], ids=["pending-output", "exhausted-mask"]
    )
    def test_ineligible_processors_fall_back_with_original_logits(
        self, monkeypatch, exhausted
    ):
        params = SamplingParams(
            temperature=0.8, presence_penalty=0.5, allowed_token_ids=[1]
        )
        outputs = [[-1]]
        if exhausted:
            params._bad_words_token_ids = [[1]]
            outputs = [[]]
            monkeypatch.setattr(
                sampling_batch,
                "apply_native_penalties",
                lambda *_: pytest.fail("penalties reached an exhausted mask"),
            )
        batch = SamplingBatch([params], [[0]], outputs, vocab_size=2)
        logits = mx.array([[2.0, 1.0]])
        sampler = Sampler()
        forward = sampler.forward
        seen = []

        def record(scores, metadata):
            seen.append(scores.tolist())
            return forward(scores, metadata)

        monkeypatch.setattr(sampler, "forward", record)
        sample_from_logits(
            logits,
            batch,
            sampler,
            next_native_rng_key=lambda: pytest.fail("fallback advanced native RNG"),
        )
        assert seen == [[[2.0, 1.0]]]

    @pytest.mark.parametrize(
        ("prompt", "output"),
        [([2], []), ([], [2]), ([3], [])],
        ids=["prompt-at-boundary", "output-at-boundary", "prompt-beyond-boundary"],
    )
    def test_out_of_vocab_history_stays_out_of_scatter(self, prompt, output):
        params = [SamplingParams(repetition_penalty=1.2)]
        assert (
            apply_native_penalties(mx.array([[0.0, 1.0]]), params, [prompt], [output])
            is None
        )


class TestNativeLogprobs:
    @pytest.fixture(
        params=["raw_logprobs", "raw_logits", "processed_logprobs", "processed_logits"]
    )
    def logprobs_mode(self, request):
        return request.param

    @pytest.fixture(
        params=[
            pytest.param({"temperature": 0.0}, id="greedy"),
            pytest.param({"temperature": 0.7}, id="random"),
            pytest.param(
                {"temperature": 0.7, "top_k": 3, "top_p": 0.85, "min_p": 0.05},
                id="filtered",
            ),
        ]
    )
    def sampling_kwargs(self, request):
        return request.param

    def _sample_with_reference(self, monkeypatch, mode, raw, batch):
        metadata = batch.make_sampling_metadata()
        metadata.max_num_logprobs = -1
        metadata.logprob_token_ids = None
        reference = Sampler(logprobs_mode=mode).forward(
            torch.tensor(raw)[:, : batch.vocab_size].contiguous(), metadata
        )
        assert reference.logprobs_tensors is not None
        expected_scores = reference.logprobs_tensors.logprobs
        sampler = Sampler(logprobs_mode=mode)
        monkeypatch.setattr(
            sampler, "forward", lambda *_: pytest.fail("torch fallback")
        )
        result = sample_from_logits(
            mx.array(raw), batch, sampler, next_native_rng_key=lambda: mx.random.key(9)
        )
        assert result.logprobs is not None
        return result, expected_scores

    def _sample_plain(self, monkeypatch, mode, raw, params):
        batch = SamplingBatch(
            params, [[1]] * len(params), [[] for _ in params], vocab_size=len(raw[0])
        )
        return self._sample_with_reference(monkeypatch, mode, raw, batch)

    def _assert_single_row_scores(self, result, reference):
        ids = result.logprobs.logprob_token_ids[0].tolist()
        selected = result.token_ids[0]
        assert ids[0] == selected
        torch.testing.assert_close(
            torch.tensor(result.logprobs.logprobs[0]),
            reference[0, ids],
            atol=2e-6,
            rtol=2e-6,
        )
        assert result.logprobs.sampled_token_ranks[0] == int(
            (reference[0] >= reference[0, selected]).sum()
        )

    def test_topk_scores_match_vllm(self, monkeypatch, logprobs_mode, sampling_kwargs):
        params = [SamplingParams(logprobs=2, **sampling_kwargs)]
        raw = [[1.0, 3.0, 2.0, -1.0, 0.0]]
        result, reference = self._sample_plain(monkeypatch, logprobs_mode, raw, params)
        assert result.logprobs.logprob_token_ids.shape == (1, 3)
        assert (
            result.logprobs.logprob_token_ids[0, 1:].tolist()
            == torch.topk(reference[0], 2).indices.tolist()
        )
        self._assert_single_row_scores(result, reference)

    def test_explicit_token_scores_match_vllm(
        self, monkeypatch, logprobs_mode, sampling_kwargs
    ):
        params = [SamplingParams(logprob_token_ids=[0, 3], **sampling_kwargs)]
        raw = [[-2.0, 0.0, 3.0, 1.0, 2.0]]
        result, reference = self._sample_plain(monkeypatch, logprobs_mode, raw, params)
        assert result.logprobs.logprob_token_ids.shape == (1, 3)
        assert result.logprobs.logprob_token_ids[0, 1:].tolist() == [0, 3]
        self._assert_single_row_scores(result, reference)

    def test_mixed_score_requests_preserve_row_alignment_and_padding(self, monkeypatch):
        params = [
            SamplingParams(temperature=0, logprobs=2),
            SamplingParams(
                temperature=1.3,
                top_k=3,
                top_p=0.85,
                min_p=0.05,
                logprob_token_ids=[0, 3],
            ),
            SamplingParams(temperature=0.5, logprobs=0),
        ]
        raw = [
            [1.0, 4.0, 2.0, -1.0, 0.0],
            [-2.0, 0.0, 3.0, 1.0, 2.0],
            [0.0, 2.0, 1.0, -3.0, 4.0],
        ]
        result, reference = self._sample_plain(monkeypatch, "raw_logprobs", raw, params)
        expected_ids = torch.tensor(
            [
                [result.token_ids[0], 1, 2],
                [result.token_ids[1], 0, 3],
                [result.token_ids[2], 0, 0],
            ]
        )
        assert result.logprobs.logprob_token_ids.shape == (3, 3)
        assert result.logprobs.logprob_token_ids.tolist() == expected_ids.tolist()
        expected_scores = reference.gather(1, expected_ids)
        expected_scores[2, 1:] = -torch.inf
        torch.testing.assert_close(
            torch.tensor(result.logprobs.logprobs),
            expected_scores,
            atol=2e-6,
            rtol=2e-6,
        )
        expected_ranks = (reference >= expected_scores[:, :1]).sum(dim=-1)
        assert result.logprobs.sampled_token_ranks.tolist() == expected_ranks.tolist()
        assert not params_allow_native_greedy(params)
        assert not params_allow_native_random(params, allow_seed=True)

    @pytest.mark.parametrize("temperature", [0.0, 0.7], ids=["greedy", "filtered"])
    def test_sample_scores_compose_with_masks_and_penalties(
        self, monkeypatch, logprobs_mode, temperature
    ):
        params = [
            SamplingParams(
                temperature=temperature,
                top_k=3,
                top_p=0.9,
                min_p=0.02,
                logprobs=2,
                allowed_token_ids=[0, 1, 2, 3, 4, 6],
                repetition_penalty=1.5,
                frequency_penalty=0.5,
                presence_penalty=0.25,
            ),
            SamplingParams(
                temperature=0,
                logprobs=1,
                logprob_token_ids=[1, 4, 6],
                allowed_token_ids=[0, 1, 2, 3, 4, 6],
                repetition_penalty=0.7,
                frequency_penalty=-0.2,
                presence_penalty=0.3,
            ),
        ]
        params[0]._bad_words_token_ids = [[5, 0]]
        params[1]._bad_words_token_ids = [[5, 1]]
        if temperature:
            # SamplingParams clears greedy filters at construction. A caller
            # can still update them before submitting a mixed batch.
            params[1].top_k = 2
            params[1].top_p = 0.6
            params[1].min_p = 0.02
        batch = SamplingBatch(
            params, [[2, 2], [6]], [[1, 1, 5], [2, 2, 5]], vocab_size=8
        )
        raw = [[9, 4, 9.6, 7, 3, 6, 2, 1, 100], [2, 7, 1, 3, 0, -2, 4, 5, 88]]
        result, reference = self._sample_with_reference(
            monkeypatch, logprobs_mode, raw, batch
        )
        assert result.token_ids[0] in {1, 2, 3, 4, 6}
        assert result.token_ids[1] == 6
        topk = torch.topk(reference[0], 2).indices.tolist()
        ids = [[result.token_ids[0], *topk, 0], [6, 1, 4, 6]]
        assert result.logprobs.logprob_token_ids.tolist() == ids
        expected = reference.gather(1, torch.tensor(ids))
        expected[0, -1] = -torch.inf
        torch.testing.assert_close(
            torch.tensor(result.logprobs.logprobs), expected, atol=2e-6, rtol=2e-6
        )
        ranks = (reference >= expected[:, :1]).sum(dim=-1)
        assert result.logprobs.sampled_token_ranks.tolist() == ranks.tolist()

    @pytest.mark.parametrize(("logits_mode", "rank"), [(True, 2), (False, 3)])
    def test_rank_uses_requested_score_space(self, logits_mode, rank):
        from vllm_metal.v1.native_logprobs import gather_native_logprobs

        raw = mx.array([[1.0, 1.0 + 2**-23, 100.0]])
        result = gather_native_logprobs(raw, mx.array([1]), 0, logits_mode=logits_mode)
        assert result.tolists().sampled_token_ranks.tolist() == [rank]

    @pytest.mark.parametrize("logits_mode", [False, True])
    def test_inclusive_ranks_ties_and_masked_targets(self, logits_mode):
        from vllm_metal.v1.native_logprobs import gather_native_logprobs

        raw = mx.array([[2.0, 2.0, 1.0, -mx.inf], [2.0, 2.0, 1.0, -mx.inf]])
        result = gather_native_logprobs(
            raw,
            mx.array([0, 3]),
            [0, 2],
            logits_mode=logits_mode,
        ).tolists()
        assert result.sampled_token_ranks.tolist() == [2, 4]
        assert result.logprob_token_ids[0].tolist() == [0, 0, 0]
        assert set(result.logprob_token_ids[1, 1:].tolist()) == {0, 1}
        assert result.logprobs[1, 0] == float("-inf")

    def test_prompt_scoring_exports_only_gathered_rows(self, monkeypatch):
        from vllm_metal.v1 import native_logprobs
        from vllm_metal.v1.prompt_logprobs import gather_prompt_logprobs

        shapes = []
        bridge = native_logprobs.mlx_to_torch

        def checked_bridge(value, **kwargs):
            shapes.append(value.shape)
            return bridge(value, **kwargs)

        monkeypatch.setattr(native_logprobs, "mlx_to_torch", checked_bridge)
        result = gather_prompt_logprobs(
            mx.arange(64, dtype=mx.float32).reshape(2, 32), [3, 4], 2
        )
        assert result.selected_token_ranks.tolist() == [29, 28]
        assert shapes == [(2, 3), (2, 3), (2,)]

    def test_large_vocabulary_retains_low_probability_mass(self):
        """FP64 reference catches loss of many tiny terms in the normalizer."""
        import math

        from vllm_metal.v1.native_logprobs import gather_native_logprobs

        vocab_size = 151936
        logits = mx.concatenate(
            [mx.zeros((1, 1)), mx.full((1, vocab_size - 1), -18.0)], axis=-1
        )
        actual = gather_native_logprobs(logits, mx.array([0]), 0).tolists()
        expected = -math.log1p((vocab_size - 1) * math.exp(-18.0))
        assert actual.logprobs[0, 0] == pytest.approx(expected, abs=5e-6)

    @pytest.mark.parametrize("logits_mode", [False, True])
    @pytest.mark.parametrize("k", [1, 3, 17])
    def test_large_topk_covers_tail_ties_and_few_finite_scores(self, logits_mode, k):
        from vllm_metal.v1.native_logprobs import gather_native_logprobs

        # A non-aligned vocabulary exercises the real tail. Maxima span chunks,
        # while one row has fewer finite scores than the requested alternatives.
        vocab = 4099
        raw = torch.stack(
            [
                torch.arange(vocab, dtype=torch.float32),
                torch.full((vocab,), -2.0),
                torch.full((vocab,), float("-inf")),
            ]
        )
        raw[1, [0, 513, 2049, vocab - 1]] = 7.0
        raw[2, 511] = 2.0
        targets = [vocab - 1, 0, vocab - 2]
        reference = raw if logits_mode else raw.log_softmax(dim=-1)
        result = gather_native_logprobs(
            mx.array(raw.tolist()),
            mx.array(targets),
            k,
            logits_mode=logits_mode,
        ).tolists()
        for row, target in enumerate(targets):
            ids = result.logprob_token_ids[row].tolist()
            assert ids[0] == target
            assert len(set(ids[1:])) == k
            assert all(0 <= token < vocab for token in ids)
            torch.testing.assert_close(
                torch.tensor(result.logprobs[row]),
                reference[row, ids],
            )
            torch.testing.assert_close(
                torch.tensor(result.logprobs[row, 1:]),
                torch.topk(reference[row], k).values,
            )
            assert result.sampled_token_ranks[row] == int(
                (reference[row] >= reference[row, target]).sum()
            )
