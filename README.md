# Bowling Simulation Engine

This project is a deterministic bowling simulation HTTP API built with Flask.

It exposes a single public URL and changes behavior based on query parameters.

## Core idea

The app behaves like a stateless function:

output = f(mode, skill, seed, isLastFrame)

The same inputs always produce the same output.

## Modes

### 1. Frame simulation

Required query params:

- `mode=frame`
- `skill=<float in [0,1]>`
- `seed=<integer>`
- `isLastFrame=<0 or 1>`

Example:

```bash
http://127.0.0.1:5000/?mode=frame&skill=0.75&seed=42&isLastFrame=0# bowling_sim
