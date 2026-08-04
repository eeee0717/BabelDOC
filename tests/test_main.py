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
    assert args.files == ["paper.pdf"]
