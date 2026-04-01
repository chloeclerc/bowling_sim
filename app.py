from __future__ import annotations

import html
import random
from dataclasses import dataclass
from typing import Any

from flask import Flask, Response, jsonify, request

app = Flask(__name__)


"""
Bowling simulation HTTP API.

Assignment-critical behavior:
- One public URL with query parameters controlling mode.
- Deterministic behavior: same inputs => same outputs.
- We seed exactly once per request with random.Random(seed).
- We never reseed inside frame simulation or during game simulation.
"""


# ----------------------------- Skill / pinfall model -----------------------------


def _clamp_skill(skill: float) -> float:
    return max(0.0, min(1.0, skill))


def _strike_probability(skill: float) -> float:
    """
    Maps skill in [0,1] to a plausible first-ball strike rate.

    Low skill still has a small strike chance.
    High skill gets a much better chance, but still far from automatic.
    """
    skill = _clamp_skill(skill)
    return 0.03 + 0.42 * skill


def _spare_conversion_probability(skill: float, pins_left: int) -> float:
    """
    Higher skill improves spare conversion.
    More standing pins makes conversion a bit harder.
    """
    skill = _clamp_skill(skill)
    base = 0.22 + 0.68 * skill
    penalty = (pins_left / 10.0) * (0.10 - 0.03 * skill)
    return max(0.05, min(0.94, base - penalty))


def roll_full_rack(rng: random.Random, skill: float) -> int:
    """
    First ball on a fresh rack of 10 pins.

    Legal outputs: 0..10
    """
    if rng.random() < _strike_probability(skill):
        return 10

    # Non-strike outcome, 0..9, biased upward as skill increases.
    # This is a simple stochastic model, not physics.
    weights: list[float] = []
    for pins in range(10):
        weights.append((pins + 1) ** (1.0 + 2.1 * skill))
    return rng.choices(population=list(range(10)), weights=weights, k=1)[0]


def roll_pickup(rng: random.Random, skill: float, pins_left: int) -> int:
    """
    Second ball on remaining pins.

    Legal outputs: 0..pins_left
    """
    if pins_left <= 0:
        return 0

    if rng.random() < _spare_conversion_probability(skill, pins_left):
        return pins_left

    if pins_left == 1:
        return 0

    return rng.randint(0, pins_left - 1)


# ----------------------------- Simulation -----------------------------


def simulate_frame(rng: random.Random, skill: float, is_last_frame: bool = False) -> list[int]:
    """
    Simulate exactly one frame.

    Frames 1-9:
    - strike ends frame immediately
    - otherwise at most two rolls

    10th frame:
    - third roll only if earned by strike or spare
    """
    rolls: list[int] = []

    if not is_last_frame:
        r1 = roll_full_rack(rng, skill)
        rolls.append(r1)

        if r1 == 10:
            return rolls

        r2 = roll_pickup(rng, skill, 10 - r1)
        rolls.append(r2)
        return rolls

    # 10th frame logic
    r1 = roll_full_rack(rng, skill)
    rolls.append(r1)

    # Strike on first ball in 10th: two fill balls.
    if r1 == 10:
        r2 = roll_full_rack(rng, skill)
        rolls.append(r2)

        if r2 == 10:
            r3 = roll_full_rack(rng, skill)
        else:
            r3 = roll_pickup(rng, skill, 10 - r2)

        rolls.append(r3)
        return rolls

    # No strike on first ball in 10th
    r2 = roll_pickup(rng, skill, 10 - r1)
    rolls.append(r2)

    # Spare earns one fill ball on a fresh rack.
    if r1 + r2 == 10:
        r3 = roll_full_rack(rng, skill)
        rolls.append(r3)

    return rolls


def simulate_game(rng: random.Random, skill: float) -> list[list[int]]:
    """
    Simulate a full 10-frame game.

    Important:
    - one RNG per whole request
    - no reseeding between frames
    """
    frames: list[list[int]] = []

    for _ in range(9):
        frames.append(simulate_frame(rng, skill, is_last_frame=False))

    frames.append(simulate_frame(rng, skill, is_last_frame=True))
    return frames


# ----------------------------- Scoring -----------------------------


@dataclass
class ScoredFrame:
    frame_number: int
    rolls: list[int]
    frame_score: int
    cumulative_score: int


def score_frames(frame_rolls: list[list[int]]) -> tuple[list[ScoredFrame], int]:
    """
    Official ten-pin scoring.

    Frames 1-9:
    - strike = 10 + next two rolls
    - spare = 10 + next one roll
    - open = sum of frame rolls

    Frame 10:
    - score is just the sum of rolls in that frame
    """
    flat_rolls: list[int] = []
    for fr in frame_rolls:
        flat_rolls.extend(fr)

    scored: list[ScoredFrame] = []
    cumulative = 0
    roll_idx = 0

    for frame_idx in range(10):
        rolls = frame_rolls[frame_idx]

        if frame_idx < 9:
            if rolls[0] == 10:
                frame_score = 10 + flat_rolls[roll_idx + 1] + flat_rolls[roll_idx + 2]
                roll_idx += 1
            elif len(rolls) >= 2 and rolls[0] + rolls[1] == 10:
                frame_score = 10 + flat_rolls[roll_idx + 2]
                roll_idx += 2
            else:
                frame_score = sum(rolls[:2])
                roll_idx += 2
        else:
            frame_score = sum(rolls)
            roll_idx += len(rolls)

        cumulative += frame_score
        scored.append(
            ScoredFrame(
                frame_number=frame_idx + 1,
                rolls=list(rolls),
                frame_score=frame_score,
                cumulative_score=cumulative,
            )
        )

    return scored, cumulative


# ----------------------------- Validation -----------------------------


def _error(message: str, status: int = 400) -> Response:
    return jsonify({"error": message}), status


def _parse_mode(raw: str | None) -> tuple[str | None, str | None]:
    if not raw:
        return None, "Missing required parameter: mode"
    if raw not in {"frame", "game", "viz"}:
        return None, "mode must be one of: frame, game, viz"
    return raw, None


def _parse_float01(name: str, raw: str | None) -> tuple[float | None, str | None]:
    if raw is None or raw == "":
        return None, f"Missing required parameter: {name}"
    try:
        value = float(raw)
    except ValueError:
        return None, f"{name} must be a number"
    if not (0.0 <= value <= 1.0):
        return None, f"{name} must be in [0, 1]"
    return value, None


def _parse_int(name: str, raw: str | None, *, required: bool) -> tuple[int | None, str | None]:
    if raw is None or raw == "":
        if required:
            return None, f"Missing required parameter: {name}"
        return None, None
    try:
        value = int(raw)
    except ValueError:
        return None, f"{name} must be an integer"
    return value, None


def _parse_is_last_frame(raw: str | None, *, required: bool) -> tuple[bool | None, str | None]:
    if raw is None or raw == "":
        if required:
            return None, "Missing required parameter: isLastFrame (required for mode=frame)"
        return None, None
    if raw == "0":
        return False, None
    if raw == "1":
        return True, None
    return None, "isLastFrame must be 0 or 1"


# ----------------------------- JSON helpers -----------------------------


def _frame_response(skill: float, seed: int, is_last: bool, rolls: list[int]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "skill": skill,
        "seed": seed,
        "is_last_frame": is_last,
        "roll1": rolls[0],
    }
    if len(rolls) >= 2:
        body["roll2"] = rolls[1]
    if len(rolls) >= 3:
        body["roll3"] = rolls[2]
    return body


# ----------------------------- Visualization helpers -----------------------------


def _mark_open_value(pins: int) -> str:
    return "-" if pins == 0 else str(pins)


def _rolls_display(rolls: list[int], frame_index: int) -> list[str]:
    """
    Convert numeric rolls to scorecard marks.

    Frames 1-9:
    - strike shown as ["", "X"] to mimic a real two-cell frame
    - spare shown with "/"

    Frame 10:
    - up to three marks
    """
    if frame_index < 9:
        if len(rolls) == 1 and rolls[0] == 10:
            return ["", "X"]

        r1, r2 = rolls[0], rolls[1]
        first = _mark_open_value(r1)

        if r1 + r2 == 10:
            second = "/"
        else:
            second = _mark_open_value(r2)

        return [first, second]

    # 10th frame
    out: list[str] = []

    r1 = rolls[0]
    out.append("X" if r1 == 10 else _mark_open_value(r1))

    if len(rolls) >= 2:
        r2 = rolls[1]
        if r1 == 10:
            out.append("X" if r2 == 10 else _mark_open_value(r2))
        else:
            out.append("/" if r1 + r2 == 10 else _mark_open_value(r2))

    if len(rolls) >= 3:
        r3 = rolls[2]
        if r1 == 10:
            if r2 == 10:
                out.append("X" if r3 == 10 else _mark_open_value(r3))
            else:
                out.append("/" if r2 + r3 == 10 else _mark_open_value(r3))
        else:
            out.append("X" if r3 == 10 else _mark_open_value(r3))

    return out


def _frame_raw_rolls_text(rolls: list[int]) -> str:
    return ", ".join(str(x) for x in rolls)


def _render_frame_box(sf: ScoredFrame) -> str:
    marks = _rolls_display(sf.rolls, sf.frame_number - 1)
    raw = html.escape(_frame_raw_rolls_text(sf.rolls))

    if sf.frame_number < 10:
        while len(marks) < 2:
            marks.insert(0, "")
        marks_html = "".join(f'<div class="mark">{html.escape(mark)}</div>' for mark in marks)
        return f"""
        <div class="frame">
          <div class="frame-number">{sf.frame_number}</div>
          <div class="marks">{marks_html}</div>
          <div class="score">{sf.cumulative_score}</div>
          <div class="frame-detail">frame score: {sf.frame_score}</div>
          <div class="frame-detail">raw: [{raw}]</div>
        </div>
        """

    while len(marks) < 3:
        marks.append("")
    marks_html = "".join(f'<div class="mark">{html.escape(mark)}</div>' for mark in marks)
    return f"""
    <div class="frame tenth">
      <div class="frame-number">{sf.frame_number}</div>
      <div class="marks marks-tenth">{marks_html}</div>
      <div class="score">{sf.cumulative_score}</div>
      <div class="frame-detail">frame score: {sf.frame_score}</div>
      <div class="frame-detail">raw: [{raw}]</div>
    </div>
    """

def _render_viz_html(skill: float, seed: int, frames: list[ScoredFrame], total: int) -> str:
    import json

    initial_game = {
        "skill": skill,
        "seed": seed,
        "total_score": total,
        "frames": [
            {
                "frame_number": sf.frame_number,
                "rolls": sf.rolls,
                "frame_score": sf.frame_score,
                "cumulative_score": sf.cumulative_score,
            }
            for sf in frames
        ],
    }

    # JSON inside a <script type="application/json"> tag should not be HTML-escaped.
    # Only neutralize the "</" sequence to avoid accidentally closing the script tag.
    initial_game_json = json.dumps(initial_game).replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bowling Simulation</title>
  <style>
    :root {{
      --bg-1: #120d08;
      --bg-2: #24170e;
      --wood-1: #8c5a2b;
      --wood-2: #d39a5c;
      --wood-3: #f2cf9e;
      --neon-red: #ff5a5f;
      --neon-blue: #6ee7ff;
      --neon-gold: #ffd166;
      --panel: rgba(28, 20, 14, 0.88);
      --panel-2: rgba(255,255,255,0.06);
      --line: rgba(255,255,255,0.14);
      --text: #fff7ed;
      --muted: #dbcbb7;
      --shadow: 0 18px 50px rgba(0,0,0,0.35);
      --radius: 24px;
    }}

    * {{
      box-sizing: border-box;
    }}

    html, body {{
      margin: 0;
      padding: 0;
      min-height: 100%;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top, rgba(255, 193, 7, 0.12), transparent 30%),
        linear-gradient(180deg, #1b130d 0%, #0f0b08 100%);
    }}

    body {{
      padding: 28px 18px 40px;
    }}

    .wrap {{
      max-width: min(1680px, 100%);
      margin: 0 auto;
    }}

    .hero {{
      position: relative;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 28px;
      background:
        linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)),
        linear-gradient(180deg, rgba(255,255,255,0.03), rgba(0,0,0,0.12));
      backdrop-filter: blur(10px);
      margin-bottom: 22px;
    }}

    .hero::before {{
      content: "";
      position: absolute;
      inset: auto -80px -80px auto;
      width: 240px;
      height: 240px;
      background: radial-gradient(circle, rgba(110, 231, 255, 0.22), transparent 70%);
      pointer-events: none;
    }}

    .hero::after {{
      content: "";
      position: absolute;
      inset: -80px auto auto -80px;
      width: 220px;
      height: 220px;
      background: radial-gradient(circle, rgba(255, 90, 95, 0.18), transparent 70%);
      pointer-events: none;
    }}

    h1 {{
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 0.98;
      letter-spacing: -0.03em;
    }}

    .subtitle {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 1.02rem;
      max-width: 760px;
      line-height: 1.45;
    }}

    .title-row {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-start;
      flex-wrap: wrap;
    }}

    .badge-row {{
      margin-top: 18px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}

    .badge {{
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(255,255,255,0.06);
      color: var(--text);
      padding: 9px 13px;
      border-radius: 999px;
      font-size: 0.92rem;
    }}

    .layout {{
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
      gap: 22px;
      align-items: start;
    }}

    .layout > main {{
      min-width: 0;
    }}

    .panel {{
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      background: var(--panel);
      overflow: hidden;
    }}

    .panel-inner {{
      padding: 22px;
    }}

    .panel-title {{
      font-size: 0.92rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--neon-gold);
      margin-bottom: 14px;
      font-weight: 700;
    }}

    .control-group {{
      margin-bottom: 18px;
    }}

    .control-label {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
      font-weight: 700;
      font-size: 0.96rem;
    }}

    .control-help {{
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.35;
      margin-top: 7px;
    }}

    input[type="range"] {{
      width: 100%;
      accent-color: #ffd166;
      cursor: pointer;
    }}

    input[type="number"], select {{
      width: 100%;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.16);
      color: var(--text);
      border-radius: 14px;
      padding: 12px 14px;
      font-size: 1rem;
      outline: none;
    }}

    .button-row {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
      margin-top: 16px;
    }}

    button {{
      border: none;
      border-radius: 16px;
      padding: 12px 14px;
      font-weight: 800;
      font-size: 0.95rem;
      cursor: pointer;
      transition: transform 0.18s ease, box-shadow 0.18s ease, opacity 0.18s ease;
    }}

    button:hover {{
      transform: translateY(-1px);
    }}

    .btn-primary {{
      background: linear-gradient(135deg, var(--neon-gold), #ff9f1c);
      color: #28170b;
      box-shadow: 0 10px 24px rgba(255, 209, 102, 0.26);
    }}

    .btn-secondary {{
      background: linear-gradient(135deg, #89f7fe, #66a6ff);
      color: #08111f;
      box-shadow: 0 10px 24px rgba(102, 166, 255, 0.24);
    }}

    .btn-dark {{
      background: rgba(255,255,255,0.08);
      color: var(--text);
      border: 1px solid rgba(255,255,255,0.14);
    }}

    .stack-buttons {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      margin-top: 10px;
    }}

    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}

    .stat {{
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
      min-width: 0;
    }}

    .stat-label {{
      color: var(--muted);
      font-size: 0.88rem;
      margin-bottom: 6px;
    }}

    .stat-value {{
      font-size: clamp(1.2rem, 2vw, 1.8rem);
      font-weight: 900;
      letter-spacing: -0.03em;
    }}

    .lane-card {{
      position: relative;
      border: 1px solid var(--line);
      border-radius: 26px;
      overflow: hidden;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)),
        linear-gradient(180deg, #4b2e16 0%, #8c5a2b 16%, #d39a5c 55%, #f2cf9e 100%);
      margin-bottom: 18px;
    }}

    .lane-top {{
      background: linear-gradient(180deg, rgba(17,24,39,0.8), rgba(17,24,39,0.45));
      padding: 16px 18px;
      border-bottom: 1px solid rgba(255,255,255,0.12);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }}

    .lane-title {{
      font-weight: 900;
      font-size: 1.05rem;
    }}

    .lane-sub {{
      color: #f6e9d8;
      opacity: 0.86;
      font-size: 0.92rem;
    }}

    .lane {{
      position: relative;
      height: 170px;
      overflow: hidden;
    }}

    .lane::before {{
      content: "";
      position: absolute;
      inset: 0;
      background:
        repeating-linear-gradient(
          90deg,
          rgba(255,255,255,0.07) 0 12px,
          rgba(0,0,0,0.04) 12px 24px
        );
      mix-blend-mode: soft-light;
      pointer-events: none;
    }}

    .lane::after {{
      content: "";
      position: absolute;
      left: 50%;
      top: 0;
      bottom: 0;
      width: 4px;
      transform: translateX(-50%);
      background: rgba(255,255,255,0.35);
      box-shadow: 0 0 12px rgba(255,255,255,0.12);
      pointer-events: none;
    }}

    .pins {{
      position: absolute;
      top: 18px;
      left: 50%;
      transform: translateX(-50%);
      display: grid;
      gap: 8px;
      justify-items: center;
    }}

    .pin-row {{
      display: flex;
      gap: 8px;
      justify-content: center;
    }}

    .pin {{
      width: 16px;
      height: 34px;
      border-radius: 10px 10px 8px 8px;
      background: linear-gradient(180deg, #fff 0%, #f8f8f8 100%);
      border: 1px solid rgba(0,0,0,0.1);
      position: relative;
      box-shadow: 0 5px 14px rgba(0,0,0,0.15);
    }}

    .pin::after {{
      content: "";
      position: absolute;
      left: 2px;
      right: 2px;
      top: 9px;
      height: 5px;
      border-radius: 999px;
      background: var(--neon-red);
    }}

    .pin.pin-down {{
      opacity: 0.28;
      transform: translateY(10px) scale(0.92);
      filter: grayscale(0.35);
    }}

    .ball {{
      position: absolute;
      bottom: 18px;
      left: 28px;
      width: 36px;
      height: 36px;
      border-radius: 50%;
      background:
        radial-gradient(circle at 30% 28%, #4fc3f7 0%, #0ea5e9 22%, #1d4ed8 70%, #0f172a 100%);
      box-shadow: 0 12px 18px rgba(0,0,0,0.26);
      transform: translateX(0) translateY(0) scale(1);
    }}

    .ball::before, .ball::after {{
      content: "";
      position: absolute;
      width: 5px;
      height: 5px;
      border-radius: 50%;
      background: rgba(1, 23, 43, 0.92);
      top: 10px;
    }}

    .ball::before {{
      left: 10px;
    }}

    .ball::after {{
      left: 18px;
    }}

    .ball-hole {{
      position: absolute;
      width: 5px;
      height: 5px;
      border-radius: 50%;
      background: rgba(1, 23, 43, 0.92);
      top: 17px;
      left: 14px;
    }}

    .ball.animate {{
      animation: rollDown 1s cubic-bezier(.2,.82,.18,1) forwards;
    }}

    @keyframes rollDown {{
      0% {{
        transform: translateX(0) translateY(0) scale(1) rotate(0deg);
      }}
      55% {{
        transform: translateX(480px) translateY(-10px) scale(0.82) rotate(320deg);
      }}
      100% {{
        transform: translateX(790px) translateY(-80px) scale(0.55) rotate(720deg);
        opacity: 0.92;
      }}
    }}

    .scoreboard {{
      border: 1px solid var(--line);
      border-radius: 26px;
      padding: 18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.025));
      overflow-x: visible;
      min-width: 0;
    }}

    .frames {{
      display: grid;
      grid-template-columns: repeat(10, minmax(0, 1fr));
      gap: clamp(6px, 1vw, 12px);
      width: 100%;
      min-width: 0;
    }}

    .frame {{
      border-radius: 18px;
      overflow: hidden;
      border: 2px solid rgba(255,255,255,0.16);
      background: linear-gradient(180deg, rgba(255,255,255,0.085), rgba(255,255,255,0.03));
      transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }}

    .frame:hover {{
      transform: translateY(-2px);
      border-color: rgba(255, 209, 102, 0.55);
      box-shadow: 0 14px 28px rgba(0,0,0,0.18);
    }}

    .frame.active {{
      border-color: rgba(110, 231, 255, 0.9);
      box-shadow: 0 0 0 2px rgba(110, 231, 255, 0.14), 0 14px 28px rgba(0,0,0,0.22);
    }}

    .frame.tenth {{
      background: linear-gradient(180deg, rgba(255, 209, 102, 0.14), rgba(255,255,255,0.04));
    }}

    .frame-head {{
      padding: clamp(6px, 1.2vw, 10px) clamp(4px, 1vw, 12px);
      border-bottom: 1px solid rgba(255,255,255,0.14);
      font-weight: 800;
      font-size: clamp(0.68rem, 1.35vw, 0.88rem);
      display: flex;
      justify-content: space-between;
      gap: 4px;
      align-items: center;
    }}

    .frame-bonus {{
      font-size: clamp(0.62rem, 1.15vw, 0.78rem);
      color: var(--neon-gold);
    }}

    .marks {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      border-bottom: 1px solid rgba(255,255,255,0.14);
    }}

    .marks.tenth {{
      grid-template-columns: repeat(3, 1fr);
    }}

    .mark {{
      min-height: clamp(36px, 7vw, 58px);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: clamp(0.95rem, 2.6vw, 1.55rem);
      font-weight: 900;
      border-right: 1px solid rgba(255,255,255,0.14);
    }}

    .mark:last-child {{
      border-right: none;
    }}

    .running-total {{
      text-align: center;
      font-size: clamp(1rem, 2.8vw, 1.9rem);
      font-weight: 900;
      padding: clamp(6px, 1.2vw, 12px) clamp(2px, 0.8vw, 10px) clamp(4px, 0.8vw, 6px);
      letter-spacing: -0.04em;
    }}

    .mini {{
      text-align: center;
      color: var(--muted);
      font-size: clamp(0.65rem, 1.5vw, 0.84rem);
      padding: 0 clamp(2px, 0.6vw, 10px) clamp(6px, 1.2vw, 12px);
      line-height: 1.25;
      word-break: break-word;
    }}

    .bottom-grid {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 18px;
      margin-top: 18px;
    }}

    .info-card {{
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03));
    }}

    .info-card h3 {{
      margin: 0 0 10px;
      font-size: 1.05rem;
    }}

    .info-card p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
      font-size: 0.94rem;
    }}

    .legend-list {{
      display: grid;
      gap: 10px;
      margin-top: 10px;
    }}

    .legend-item {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(255,255,255,0.05);
    }}

    .legend-key {{
      font-weight: 900;
      color: var(--neon-blue);
    }}

    .status {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 0.88rem;
      min-height: 1.2em;
    }}

    .toggle-row {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 10px;
    }}

    .chip {{
      padding: 9px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.12);
      font-size: 0.88rem;
      cursor: pointer;
      user-select: none;
    }}

    .chip.active {{
      background: rgba(255, 209, 102, 0.18);
      border-color: rgba(255, 209, 102, 0.42);
      color: #fff6d9;
    }}

    @media (max-width: 1120px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}

      .stats {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}

      .bottom-grid {{
        grid-template-columns: 1fr;
      }}
    }}

    @media (max-width: 800px) {{
      .scoreboard {{
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
      }}

      .frames {{
        min-width: 560px;
      }}
    }}

    @media (max-width: 640px) {{
      .stats {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="title-row">
        <div>
          <h1>Bowling Alley Simulator</h1>
          <div class="subtitle">
            Tweak skill, change the seed, reroll the game, and watch the same deterministic engine power a more fun scoreboard.
            Same inputs still give the same game.
          </div>
        </div>
      </div>

      <div class="badge-row">
        <div class="badge">Single URL</div>
        <div class="badge">Deterministic by seed</div>
        <div class="badge">Official strike / spare scoring</div>
        <div class="badge">Interactive viz mode</div>
      </div>
    </section>

    <div class="layout">
      <aside class="panel">
        <div class="panel-inner">
          <div class="panel-title">Controls</div>

          <div class="control-group">
            <div class="control-label">
              <span>Skill</span>
              <span id="skillValue">{skill:.2f}</span>
            </div>
            <input id="skillSlider" type="range" min="0" max="1" step="0.01" value="{skill:.2f}">
            <div class="control-help">
              Higher skill improves strike rate, first-ball pinfall, and spare conversion.
            </div>
          </div>

          <div class="control-group">
            <div class="control-label">
              <span>Seed</span>
              <span>integer</span>
            </div>
            <input id="seedInput" type="number" step="1" value="{seed}">
            <div class="control-help">
              Same skill and same seed always reproduce the same simulated game.
            </div>
          </div>

          <div class="button-row">
            <button class="btn-primary" id="rerollBtn">Reroll game</button>
          </div>

          <div class="stack-buttons">
            <button class="btn-dark" id="randomSeedBtn">Random seed</button>
            <button class="btn-dark" id="copyLinkBtn">Copy shareable URL</button>
          </div>

          <div class="panel-title" style="margin-top: 22px;">Extras</div>

          <div class="toggle-row">
            <div class="chip active" id="chipRaw">Show raw rolls</div>
            <div class="chip active" id="chipHighlight">Highlight best frame</div>
          </div>

          <div class="status" id="statusText"></div>
        </div>
      </aside>

      <main>
        <div class="stats">
          <div class="stat">
            <div class="stat-label">Total score</div>
            <div class="stat-value" id="totalScore">{total}</div>
          </div>
          <div class="stat">
            <div class="stat-label">Strikes</div>
            <div class="stat-value" id="strikeCount">-</div>
          </div>
          <div class="stat">
            <div class="stat-label">Spares</div>
            <div class="stat-value" id="spareCount">-</div>
          </div>
          <div class="stat">
            <div class="stat-label">Open frames</div>
            <div class="stat-value" id="openCount">-</div>
          </div>
        </div>

        <section class="lane-card">
          <div class="lane-top">
            <div>
              <div class="lane-title">Neon Lane Preview</div>
              <div class="lane-sub" id="laneSub">Frame preview updates from the live controls.</div>
            </div>
            <div class="badge">Theme: bowling alley</div>
          </div>
          <div class="lane">
            <div class="pins">
              <div class="pin-row"><div class="pin"></div></div>
              <div class="pin-row"><div class="pin"></div><div class="pin"></div></div>
              <div class="pin-row"><div class="pin"></div><div class="pin"></div><div class="pin"></div></div>
              <div class="pin-row"><div class="pin"></div><div class="pin"></div><div class="pin"></div><div class="pin"></div></div>
            </div>
            <div class="ball" id="bowlingBall"><div class="ball-hole"></div></div>
          </div>
        </section>

        <section class="scoreboard">
          <div class="frames" id="framesGrid"></div>

          <div class="bottom-grid">
            <div class="info-card">
              <h3>Frame preview</h3>
              <p id="framePreviewText">
                Hover or tap a frame to inspect its marks, raw rolls, frame score, and running total.
              </p>
            </div>

            <div class="info-card">
              <h3>Legend</h3>
              <div class="legend-list">
                <div class="legend-item"><span class="legend-key">X</span><span>strike</span></div>
                <div class="legend-item"><span class="legend-key">/</span><span>spare</span></div>
                <div class="legend-item"><span class="legend-key">-</span><span>0 pins</span></div>
                <div class="legend-item"><span class="legend-key">seed</span><span>controls reproducibility</span></div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  </div>

  <script id="initial-game-data" type="application/json">{initial_game_json}</script>
  <script>
    const initialGame = JSON.parse(document.getElementById("initial-game-data").textContent);

    const skillSlider = document.getElementById("skillSlider");
    const skillValue = document.getElementById("skillValue");
    const seedInput = document.getElementById("seedInput");
    const rerollBtn = document.getElementById("rerollBtn");
    const randomSeedBtn = document.getElementById("randomSeedBtn");
    const copyLinkBtn = document.getElementById("copyLinkBtn");
    const statusText = document.getElementById("statusText");
    const framesGrid = document.getElementById("framesGrid");
    const totalScore = document.getElementById("totalScore");
    const strikeCount = document.getElementById("strikeCount");
    const spareCount = document.getElementById("spareCount");
    const openCount = document.getElementById("openCount");
    const framePreviewText = document.getElementById("framePreviewText");
    const bowlingBall = document.getElementById("bowlingBall");
    const laneSub = document.getElementById("laneSub");
    const chipRaw = document.getElementById("chipRaw");
    const chipHighlight = document.getElementById("chipHighlight");

    let currentGame = initialGame;
    let showRaw = true;
    let highlightBest = true;

    function clampSkill(v) {{
      const n = Number(v);
      if (Number.isNaN(n)) return 0.5;
      return Math.max(0, Math.min(1, n));
    }}

    function cleanSeed(v) {{
      const n = parseInt(v, 10);
      return Number.isNaN(n) ? 42 : n;
    }}

    function setStatus(msg) {{
      statusText.textContent = msg;
    }}

    function markOpenValue(x) {{
      return x === 0 ? "-" : String(x);
    }}

    function rollsDisplay(rolls, frameIndex) {{
      if (frameIndex < 9) {{
        if (rolls.length === 1 && rolls[0] === 10) {{
          return ["", "X"];
        }}
        const r1 = rolls[0];
        const r2 = rolls[1];
        const first = markOpenValue(r1);
        const second = (r1 + r2 === 10) ? "/" : markOpenValue(r2);
        return [first, second];
      }}

      const out = [];
      const r1 = rolls[0];
      out.push(r1 === 10 ? "X" : markOpenValue(r1));

      if (rolls.length >= 2) {{
        const r2 = rolls[1];
        if (r1 === 10) {{
          out.push(r2 === 10 ? "X" : markOpenValue(r2));
        }} else {{
          out.push((r1 + r2 === 10) ? "/" : markOpenValue(r2));
        }}
      }}

      if (rolls.length >= 3) {{
        const r2 = rolls[1];
        const r3 = rolls[2];
        if (r1 === 10) {{
          if (r2 === 10) {{
            out.push(r3 === 10 ? "X" : markOpenValue(r3));
          }} else {{
            out.push((r2 + r3 === 10) ? "/" : markOpenValue(r3));
          }}
        }} else {{
          out.push(r3 === 10 ? "X" : markOpenValue(r3));
        }}
      }}

      return out;
    }}

    function countStats(game) {{
      let strikes = 0;
      let spares = 0;
      let opens = 0;

      game.frames.forEach((frame, idx) => {{
        const rolls = frame.rolls;

        if (idx < 9) {{
          if (rolls.length === 1 && rolls[0] === 10) {{
            strikes += 1;
          }} else if (rolls.length >= 2 && rolls[0] + rolls[1] === 10) {{
            spares += 1;
          }} else {{
            opens += 1;
          }}
        }} else {{
          if (rolls[0] === 10) strikes += 1;
          else if (rolls[0] + rolls[1] === 10) spares += 1;
          else opens += 1;
        }}
      }});

      return {{ strikes, spares, opens }};
    }}

    function getBestFrameIndex(game) {{
      let bestIdx = 0;
      let bestScore = -Infinity;
      game.frames.forEach((f, idx) => {{
        if (f.frame_score > bestScore) {{
          bestScore = f.frame_score;
          bestIdx = idx;
        }}
      }});
      return bestIdx;
    }}

    function pinsStandingAfterFrame(rolls, frameIndex) {{
      if (frameIndex < 9) {{
        if (rolls.length >= 1 && rolls[0] === 10) return 0;
        if (rolls.length >= 2) return Math.max(0, 10 - rolls[0] - rolls[1]);
        return Math.max(0, 10 - (rolls[0] || 0));
      }}
      if (rolls.length === 2) return Math.max(0, 10 - rolls[0] - rolls[1]);
      return 0;
    }}

    function updatePinsVisual(standing) {{
      const pins = document.querySelectorAll(".pins .pin");
      const s = Math.max(0, Math.min(10, standing));
      const down = 10 - s;
      pins.forEach((el, i) => {{
        el.classList.toggle("pin-down", i < down);
      }});
    }}

    function showFramePreview(frame, idx) {{
      laneSub.textContent =
        `Previewing frame ${{frame.frame_number}}: rolls [${{frame.rolls.join(", ")}}], frame score ${{frame.frame_score}}, running total ${{frame.cumulative_score}}.`;
      framePreviewText.textContent =
        `Frame ${{frame.frame_number}}: marks ${{rollsDisplay(frame.rolls, idx).join(" | ")}}. Raw rolls [${{frame.rolls.join(", ")}}]. ` +
        `Frame score ${{frame.frame_score}}. Running total ${{frame.cumulative_score}}.`;
      updatePinsVisual(pinsStandingAfterFrame(frame.rolls, idx));
    }}

    function renderFrames(game) {{
      framesGrid.innerHTML = "";
      const bestIdx = getBestFrameIndex(game);

      game.frames.forEach((frame, idx) => {{
        const isTenth = idx === 9;
        const marks = rollsDisplay(frame.rolls, idx);
        const container = document.createElement("div");
        container.className = "frame" + (isTenth ? " tenth" : "");
        if (highlightBest && idx === bestIdx) {{
          container.classList.add("active");
        }}

        const head = document.createElement("div");
        head.className = "frame-head";
        head.innerHTML = `<span>Frame ${{frame.frame_number}}</span><span class="frame-bonus">score ${{frame.frame_score}}</span>`;        container.appendChild(head);

        const marksWrap = document.createElement("div");
        marksWrap.className = "marks" + (isTenth ? " tenth" : "");

        const targetCells = isTenth ? 3 : 2;
        while (marks.length < targetCells) {{
          if (isTenth) marks.push("");
          else marks.unshift("");
        }}

        marks.forEach(mark => {{
          const cell = document.createElement("div");
          cell.className = "mark";
          cell.textContent = mark;
          marksWrap.appendChild(cell);
        }});

        const running = document.createElement("div");
        running.className = "running-total";
        running.textContent = frame.cumulative_score;

        const mini = document.createElement("div");
        mini.className = "mini";
        mini.textContent = showRaw ? `raw: [${{frame.rolls.join(", ")}}]` : `rolls hidden`;

        container.appendChild(marksWrap);
        container.appendChild(running);
        container.appendChild(mini);

        const preview = () => showFramePreview(frame, idx);

        container.addEventListener("mouseenter", preview);
        container.addEventListener("click", preview);

        framesGrid.appendChild(container);
      }});
    }}

    function renderStats(game) {{
      totalScore.textContent = game.total_score;
      const stats = countStats(game);
      strikeCount.textContent = stats.strikes;
      spareCount.textContent = stats.spares;
      openCount.textContent = stats.opens;
    }}

    function updateUrl(skill, seed) {{
      const url = new URL(window.location.href);
      url.searchParams.set("mode", "viz");
      url.searchParams.set("skill", skill.toFixed(2));
      url.searchParams.set("seed", String(seed));
      window.history.replaceState(null, "", url.toString());
    }}

    async function fetchGame() {{
      const skill = clampSkill(skillSlider.value);
      const seed = cleanSeed(seedInput.value);

      skillValue.textContent = skill.toFixed(2);
      updateUrl(skill, seed);
      setStatus("Loading game...");

      const params = new URLSearchParams({{
        mode: "game",
        skill: skill.toFixed(2),
        seed: String(seed),
      }});

      const response = await fetch(`/?${{params.toString()}}`);
      if (!response.ok) {{
        let msg = "Failed to load game.";
        try {{
          const err = await response.json();
          if (err.error) msg = err.error;
        }} catch (e) {{}}
        throw new Error(msg);
      }}

      const data = await response.json();
      currentGame = data;
      renderAll();
      setStatus(`Loaded deterministic game for skill=${{skill.toFixed(2)}} and seed=${{seed}}.`);
    }}

    function animateBall() {{
      bowlingBall.classList.remove("animate");
      void bowlingBall.offsetWidth;
      bowlingBall.classList.add("animate");
      setTimeout(() => bowlingBall.classList.remove("animate"), 1100);
    }}

    function renderAll() {{
      renderStats(currentGame);
      renderFrames(currentGame);
      const best = getBestFrameIndex(currentGame);
      showFramePreview(currentGame.frames[best], best);
    }}

    skillSlider.addEventListener("input", () => {{
      skillValue.textContent = clampSkill(skillSlider.value).toFixed(2);
    }});

    rerollBtn.addEventListener("click", async () => {{
      try {{
        animateBall();
        await fetchGame();
      }} catch (err) {{
        setStatus(err.message);
      }}
    }});

    randomSeedBtn.addEventListener("click", () => {{
      seedInput.value = Math.floor(Math.random() * 100000);
      setStatus("Random seed generated. Click Reroll game to load it.");
    }});

    copyLinkBtn.addEventListener("click", async () => {{
      try {{
        await navigator.clipboard.writeText(window.location.href);
        setStatus("Shareable URL copied.");
      }} catch (err) {{
        setStatus("Could not copy URL in this browser.");
      }}
    }});

    chipRaw.addEventListener("click", () => {{
      showRaw = !showRaw;
      chipRaw.classList.toggle("active", showRaw);
      renderFrames(currentGame);
      const best = getBestFrameIndex(currentGame);
      showFramePreview(currentGame.frames[best], best);
    }});

    chipHighlight.addEventListener("click", () => {{
      highlightBest = !highlightBest;
      chipHighlight.classList.toggle("active", highlightBest);
      renderFrames(currentGame);
      const best = getBestFrameIndex(currentGame);
      showFramePreview(currentGame.frames[best], best);
    }});

    seedInput.addEventListener("keydown", (e) => {{
      if (e.key === "Enter") rerollBtn.click();
    }});

    renderAll();
    setStatus("Ready. Adjust skill/seed, then click Reroll game to simulate.");
  </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def bowling_api() -> Response:
    """
    One public endpoint. Behavior depends on query parameters.

    Determinism:
    - exactly one random.Random(seed) is created per request
    - that RNG is passed through the whole simulation path
    - same mode/skill/seed inputs produce identical outputs
    """
    mode, err = _parse_mode(request.args.get("mode"))
    if err:
        return _error(err)

    skill, err = _parse_float01("skill", request.args.get("skill"))
    if err:
        return _error(err)

    seed, err = _parse_int("seed", request.args.get("seed"), required=True)
    if err:
        return _error(err)

    rng = random.Random(seed)

    if mode == "frame":
        is_last, err = _parse_is_last_frame(request.args.get("isLastFrame"), required=True)
        if err:
            return _error(err)

        rolls = simulate_frame(rng, skill, is_last_frame=is_last)
        return jsonify(_frame_response(skill, seed, is_last, rolls))

    if mode == "game":
        frame_rolls = simulate_game(rng, skill)
        scored, total = score_frames(frame_rolls)
        return jsonify(
            {
                "skill": skill,
                "seed": seed,
                "total_score": total,
                "frames": [
                    {
                        "frame_number": sf.frame_number,
                        "rolls": sf.rolls,
                        "frame_score": sf.frame_score,
                        "cumulative_score": sf.cumulative_score,
                    }
                    for sf in scored
                ],
            }
        )

    # mode == "viz"
    frame_rolls = simulate_game(rng, skill)
    scored, total = score_frames(frame_rolls)
    html_out = _render_viz_html(skill, seed, scored, total)
    return Response(html_out, mimetype="text/html; charset=utf-8")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)