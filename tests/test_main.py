from babeldoc.main import create_parser


def test_parser_exposes_json_progress_mode():
    args = create_parser().parse_args(
        [
            "--progress-json",
            "--files",
            "paper.pdf",
            "--openai",
            "--openai-api-key",
            "test",
        ]
    )

    assert args.progress_json is True
    assert args.progress_json_version == 1
    assert args.files == ["paper.pdf"]


def test_parser_accepts_json_progress_v2():
    args = create_parser().parse_args(
        [
            "--progress-json",
            "--progress-json-version",
            "2",
            "--files",
            "paper.pdf",
            "--openai",
            "--openai-api-key",
            "test",
        ]
    )

    assert args.progress_json_version == 2
