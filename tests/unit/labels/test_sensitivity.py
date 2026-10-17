from collections.abc import Callable

from dataset.cleaning import CleanDataset
from labels.generator import LabelConfig
from labels.sensitivity import label_sensitivity

type MakeClean = Callable[..., CleanDataset]


def test_one_row_per_parameter_value_and_a_stricter_threshold_keeps_fewer_turns(
    make_clean: MakeClean,
) -> None:
    closes = [100.0 + 0.001 * i for i in range(60)]
    closes[25] -= 1.0
    closes[40] += 5.0
    variations = {"threshold_coefficient": (0.5, 2.0), "min_separation_bars": (2, 4)}

    rows = label_sensitivity(make_clean(closes), LabelConfig(), variations)

    assert [(row.parameter, row.value) for row in rows] == [
        ("threshold_coefficient", 0.5),
        ("threshold_coefficient", 2.0),
        ("min_separation_bars", 2),
        ("min_separation_bars", 4),
    ]
    loose, strict = rows[0].summary.turn_event, rows[1].summary.turn_event
    assert loose["TURN_UP"] + loose["TURN_DOWN"] > strict["TURN_UP"] + strict["TURN_DOWN"]
