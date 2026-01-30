from ratchet_gpu.semantics import generate_phrase_schedule


def test_phase20_phrase_schedule_alternating() -> None:
    tokens = generate_phrase_schedule("alternating", 8, token_hold_windows=1, phrase_start=0)
    assert tokens == ["OUT", "IN", "OUT", "IN", "OUT", "IN", "OUT", "IN"]

    tokens_hold = generate_phrase_schedule("alternating", 8, token_hold_windows=2, phrase_start=0)
    assert tokens_hold == ["OUT", "OUT", "IN", "IN", "OUT", "OUT", "IN", "IN"]

    tokens_shift = generate_phrase_schedule("alternating", 4, token_hold_windows=1, phrase_start=1)
    assert tokens_shift == ["IN", "OUT", "IN", "OUT"]


def test_phase20_phrase_schedule_chunked() -> None:
    tokens = generate_phrase_schedule("chunked", 8, token_hold_windows=1, phrase_start=0)
    assert tokens == ["OUT", "OUT", "OUT", "OUT", "IN", "IN", "IN", "IN"]

    tokens_rev = generate_phrase_schedule("chunked", 8, token_hold_windows=1, phrase_start=1)
    assert tokens_rev == ["IN", "IN", "IN", "IN", "OUT", "OUT", "OUT", "OUT"]
