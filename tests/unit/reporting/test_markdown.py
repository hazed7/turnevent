from reporting.markdown import cell, document, moment, percent, section, table


def test_table_renders_github_markdown() -> None:
    assert table(["a", "b"], [[1, 0.5], ["x", None]]) == (
        "| a | b |\n| --- | --- |\n| 1 | 0.500 |\n| x | — |"
    )


def test_cells_and_percentages() -> None:
    assert cell(None) == "—"
    assert cell(0.12345) == "0.123"
    assert percent(0.265) == "26.5%"
    assert percent(None) == "—"


def test_moments_are_utc() -> None:
    assert moment(1_704_067_200_000) == "2024-01-01 00:00 UTC"


def test_documents_and_sections_stack_blocks() -> None:
    assert document("T", section("S", "body")) == "# T\n\n## S\n\nbody\n"
