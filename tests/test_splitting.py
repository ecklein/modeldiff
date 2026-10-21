from __future__ import annotations

import numpy as np
import pytest

from modeldiff import make_split


def test_random_split_matches_original_deterministic_behavior() -> None:
    split = make_split(20, strategy="random", discovery_fraction=0.4, random_state=7)
    expected = np.random.default_rng(7).permutation(20)

    np.testing.assert_array_equal(split.discovery_indices, expected[:8])
    np.testing.assert_array_equal(split.validation_indices, expected[8:])
    assert split.to_dict()["strategy"] == "random"
    assert split.to_dict()["actual_discovery_fraction"] == pytest.approx(0.4)


def test_group_split_is_deterministic_and_never_splits_a_group() -> None:
    groups = np.repeat(["a", "b", "c", "d", "e"], [4, 7, 3, 5, 6])
    first = make_split(25, strategy="group", groups=groups, random_state=19)
    second = make_split(25, strategy="group", groups=groups, random_state=19)

    np.testing.assert_array_equal(first.discovery_indices, second.discovery_indices)
    discovery_groups = set(groups[first.discovery_indices])
    validation_groups = set(groups[first.validation_indices])
    assert discovery_groups.isdisjoint(validation_groups)
    assert discovery_groups | validation_groups == set(groups)
    assert first.to_dict()["group_count"] == 5


def test_group_split_validates_group_values() -> None:
    with pytest.raises(ValueError, match="exactly 6"):
        make_split(6, strategy="group", groups=[1, 1, 2])
    with pytest.raises(ValueError, match="missing"):
        make_split(4, strategy="group", groups=[1, 1, None, 2])
    with pytest.raises(ValueError, match="two distinct groups"):
        make_split(4, strategy="group", groups=[1, 1, 1, 1])


def test_time_split_is_chronological_and_keeps_ties_together() -> None:
    times = np.array([4, 1, 3, 2, 3, 4, 5, 1])
    split = make_split(8, strategy="time", time_values=times, discovery_fraction=0.5)

    discovery_times = times[split.discovery_indices]
    validation_times = times[split.validation_indices]
    assert discovery_times.max() < validation_times.min()
    assert set(discovery_times).isdisjoint(validation_times)
    assert split.to_dict()["time_order"] == "ascending"


def test_time_split_validates_ordering_and_missing_values() -> None:
    with pytest.raises(ValueError, match="missing"):
        make_split(4, strategy="time", time_values=[1, 2, np.nan, 4])
    with pytest.raises(ValueError, match="mutually orderable"):
        make_split(4, strategy="time", time_values=[1, "two", 3, 4])
    with pytest.raises(ValueError, match="two distinct"):
        make_split(4, strategy="time", time_values=[1, 1, 1, 1])


def test_explicit_split_accepts_indices_and_boolean_masks() -> None:
    indexed = make_split(
        6,
        strategy="explicit",
        discovery_indices=[4, 0, 2],
        validation_indices=[1, 3, 5],
    )
    masked = make_split(
        6,
        strategy="explicit",
        discovery_indices=[True, False, True, False, True, False],
        validation_indices=[False, True, False, True, False, True],
    )

    np.testing.assert_array_equal(indexed.discovery_indices, [4, 0, 2])
    np.testing.assert_array_equal(masked.discovery_indices, [0, 2, 4])
    assert indexed.to_dict()["requested_discovery_fraction"] is None


@pytest.mark.parametrize(
    ("discovery", "validation", "message"),
    [
        ([0, 1], [1, 2, 3], "cannot overlap"),
        ([0, 0], [1, 2, 3], "duplicate"),
        ([0, 1], [2], "assign every observation"),
        ([0, 1], [2, 4], "outside the range"),
        ([0.0, 1.0], [2, 3], "integer indices"),
    ],
)
def test_explicit_split_rejects_invalid_partitions(
    discovery: list[object], validation: list[object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_split(
            4,
            strategy="explicit",
            discovery_indices=discovery,
            validation_indices=validation,
        )


def test_split_rejects_parameters_for_another_strategy() -> None:
    with pytest.raises(ValueError, match="groups cannot be used"):
        make_split(10, strategy="random", groups=np.arange(10))


def test_split_warns_for_small_and_unbalanced_partitions() -> None:
    groups = ["large"] * 90 + ["small"] * 10
    split = make_split(100, strategy="group", groups=groups, discovery_fraction=0.5)

    assert any("fewer than 30" in warning for warning in split.warnings)
    assert any("10 percentage points" in warning for warning in split.warnings)
