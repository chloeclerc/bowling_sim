import random

from app import score_frames, simulate_frame, simulate_game


def test_known_scoring_example():
    """
    Manual check:
    Frame 1: spare (3,7) + next roll 10 => 20
    Frame 2: strike + next two rolls 8,0 => 18
    Frame 3: open 8,0 => 8
    Remaining frames all gutters.
    Total = 20 + 18 + 8 = 46
    """
    frames = [
        [3, 7],
        [10],
        [8, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]

    scored, total = score_frames(frames)

    assert scored[0].frame_score == 20
    assert scored[1].frame_score == 18
    assert scored[2].frame_score == 8
    assert total == 46


def test_same_seed_same_game():
    rng1 = random.Random(42)
    rng2 = random.Random(42)

    g1 = simulate_game(rng1, 0.75)
    g2 = simulate_game(rng2, 0.75)

    assert g1 == g2


def test_different_seed_changes_game():
    rng1 = random.Random(42)
    rng2 = random.Random(43)

    g1 = simulate_game(rng1, 0.75)
    g2 = simulate_game(rng2, 0.75)

    assert g1 != g2


def test_non_last_frame_never_has_roll3():
    rng = random.Random(123)
    frame = simulate_frame(rng, 0.8, is_last_frame=False)
    assert len(frame) <= 2


def test_game_has_ten_frames():
    rng = random.Random(42)
    game = simulate_game(rng, 0.75)
    assert len(game) == 10