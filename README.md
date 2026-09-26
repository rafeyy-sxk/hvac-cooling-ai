# hvac-cooling-ai

A physics model of an indirect evaporative cooler, a fast ML surrogate trained on it, and a
tool-using AI assistant that answers an HVAC contractor's sizing questions by calling them.

It builds on an experimental research project on solar desiccant air cooling that I co-authored.
That project modelled a crossflow M-cycle indirect evaporative cooler in MATLAB. This repo is a
clean Python port of that model type, checked honestly against the project's published numbers,
and then wrapped in the layers a real product would need: a fast surrogate, a tool API, and an
assistant that has to show its work.

## Why

Contractors in hot, dry climates are often asked whether an evaporative or hybrid system can
replace or offload a compressor. Indirect evaporative coolers cool air with water evaporation
and a fan, and they do not add moisture to the supply air. The questions contractors get are
practical: *will it hold 26 C supply air on a 44 C afternoon? how many units for this load?*
Answering them needs psychrometrics and a heat-exchanger model, which is exactly where a
language model on its own will make numbers up. Here the model only talks; the numbers come from
tools, and the answer is thrown away if it does not cite them.

## Architecture

```
 contractor question (plain English)
            |
            v
 +-----------------------+     tool calls      +------------------------------------------+
 | agent/loop.py         | ------------------> | agent/tools.py  (plain Python, no key)   |
 | Anthropic Messages API|                     |  psychrometric_lookup  -> physics/psychro |
 | manual tool-use loop  | <------------------ |  simulate_cooler       -> physics/iec     |
 | model: claude-sonnet-5|   results R1, R2... |  predict_cooler_fast   -> surrogate/model |
 +-----------------------+                     |  units_needed          -> physics/iec     |
            |                                  |  every call: input checks + envelope gate |
            v                                  +------------------------------------------+
 +-----------------------+                                       |
 | agent/guard.py        |   physics/iec.py: 2-D cell-by-cell epsilon-NTU crossflow model
 | - must cite [R#]      |   physics/psychro.py: ASHRAE psychrometrics
 | - cited ids must be ok|   surrogate/: sweep physics -> dataset -> gradient boosting
 | - outside envelope => |   envelope.py: the validated operating range (shared by all layers)
 |   refusal, no guess   |
 +-----------------------+
            |
            v
     answer with citations, or a refusal that says why
```

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m hvac_cooling_ai simulate --t 44 --rh 30          # physics model
python -m hvac_cooling_ai predict  --t 44 --rh 30          # surrogate (milliseconds)
python -m hvac_cooling_ai size --t 40 --rh 15 --load-kw 12 --velocity 0.5 --r 0.5
python -m hvac_cooling_ai psychro  --t 44 --rh 30
python -m hvac_cooling_ai validate                         # comparison with the source project
ANTHROPIC_API_KEY=... python -m hvac_cooling_ai ask "Will this cooler hold 26 C supply air on a 44 C, 30% RH afternoon?" --show-tools

pytest -q          # 55 tests; the one live-API test is skipped without a key
ruff check . && ruff format --check .
python -m hvac_cooling_ai.surrogate.train                  # regenerate the surrogate (about 1 minute)
```

## The physics model

`physics/iec.py`: a crossflow regenerative (M-cycle style) indirect evaporative cooler.
Product air flows through a dry channel along x and is cooled sensibly through a thin plate. At
the dry outlet a fraction `r` is turned into the wet channel, which runs along y, and picks up
heat and moisture from a wetted liner. The plate is split into a 20 x 20 grid; each cell is
solved as a small crossflow exchanger with the epsilon-NTU method, and each cell's outlet is the
next cell's inlet along both flow paths. Because the wet inlet is the dry outlet, the whole
march is repeated until that coupling converges.

The wet side uses the enthalpy-potential form that follows from a Lewis factor of 1: heat flux
is proportional to the difference between saturated-air enthalpy at the wall and the air's
enthalpy. Linearising saturated enthalpy per cell turns the wet stream into an equivalent
sensible stream, so the standard crossflow epsilon-NTU relation applies. Humidity moves on the
straight line toward the wall state; if that overshoots saturation, the state is moved back to
saturation at constant enthalpy so energy is still conserved.

### Assumptions: what the source stated vs what we chose

| Item | Source project | This port |
|---|---|---|
| Crossflow, M-cycle principle, dry and wet channels separated by a plate | stated | same |
| Lewis factor = 1, fully wetted surface | stated | same |
| Constant air properties, insulated outer walls, no leakage between channels | stated | same |
| Cell-by-cell epsilon-NTU on a 2-D grid | stated | same, 20 x 20 grid |
| Solved for one channel | stated | one repeating channel pair (dry channel exchanging through both walls) |
| How working air reaches the wet channel | not stated in detail | a fraction `r` of the cooled product air is turned into the wet channel |
| Working-air fraction | not stated | 1/3 for validation; 0.2 to 0.6 in the envelope |
| Plate size | not stated | 200 x 200 mm, set by the project's wicking test (liner stays wet to about 200 mm) |
| Channel gap, plate, liner | not stated | 4 mm gap, 0.2 mm aluminium plate, 0.2 mm wet liner (the wall resistance is negligible either way) |
| Heat transfer coefficient | not stated | laminar parallel plates, Nu = 7.54 (all envelope points are laminar, Re < 1600) |
| Inlet humidity | not stated | 12.91 g/kg, the source project's operating point (see Validation) |
| Air flow | not stated | 0.630 W/K of supply air, the source project's operating point (see Validation) |
| Fan electrical input | 0.5 W | 0.5 W per channel pair, used only for COP and EER |
| Wet channel inlet "saturated", wall temperature constant | stated as boundary conditions | not used: wall temperature and wet inlet are computed, not fixed |

Psychrometrics (`physics/psychro.py`) follow ASHRAE Fundamentals 2017 chapter 1 and are tested
against PsychroLib, an independent ASHRAE implementation (saturation pressure to 1e-6 relative,
wet bulb and dew point to 0.01 K).

Grid sensitivity at 40 C, 12.9 g/kg, 1.0 m/s, r = 1/3 (supply temperature, C):
10 x 10: 28.221, 20 x 20: 28.177, 40 x 40: 28.142, 80 x 80: 28.117. The default grid is within
0.06 K of the 80 x 80 grid.

## Validation

Command: `python -m hvac_cooling_ai validate`

Operating points follow the source project: inlet air at 306 K and 321 K, inlet humidity ratio
12.91 g/kg (dew point 291.12 K), supply air 0.630 W/K (6.11e-4 kg/s, a dry-channel velocity of
0.99 m/s in this geometry), working-air fraction 1/3, fan input 0.5 W per channel pair.

| Metric | @306 K | @321 K |
|---|---|---|
| Supply (outlet) temp, K | 298.28 | 304.29 |
| Temperature drop, K | 7.72 | 16.71 |
| Cooling capacity, W | 4.86 | 10.52 |
| Wet-bulb effectiveness | 0.75 | 0.78 |
| Dew-point effectiveness | 0.52 | 0.56 |
| COP (0.5 W fan) | 9.72 | 21.04 |
| EER, Btu/h per W (COP x 3.412) | 33.16 | 71.80 |
| Supply temp with a 20x larger exchanger, K | 292.27 | 292.66 |

**Physics checks, all in `tests/`:**

1. **The dew point is a hard floor.** A regenerative cooler whose working air is its own product
   air cannot cool below the inlet dew point, so dew-point effectiveness stays at or below 1.
   Even with an exchanger 20 times larger in each direction the model stops at 292.3 to 292.7 K,
   just above the 291.1 K floor.
2. **Energy balance closes** between the dry and wet streams in every run.
3. **EER uses the standard conversion,** COP x 3.412 Btu/h per W.
4. **COP depends on the fan figure.** COP here is cooling divided by 0.5 W. The friction pressure
   drop in these channels is 2.74 Pa (dry) and 0.91 Pa (wet), so the ideal power to move the air is
   about 2.4 mW per channel; 0.5 W stands in for real fan, pump and control losses, and COP moves
   with it.

## Surrogate

Command: `python -m hvac_cooling_ai.surrogate.train` (writes `surrogate/artifacts/`).

* **Data.** 6000 points sampled uniformly in the envelope (inlet 25 to 50 C, 4 to 20 g/kg,
  RH at most 90%, channel velocity 0.5 to 3.0 m/s, working-air fraction 0.2 to 0.6), each
  labelled by the physics model. Labelling took 18.5 s on a laptop (batched).
* **Targets.** Wet-bulb effectiveness (converted to supply temperature) and evaporated water
  per kg of supply air. Both are independent of how many channel pairs are stacked.
* **Split without leakage.** The envelope is cut into 300 blocks (5 x 5 x 4 x 3). 60 whole blocks
  (1231 points) are held out as the test set; a further 20% of the remaining blocks is the
  validation set used to choose the model. No test block is ever trained on (asserted in code).
* **Model choice on validation.** Gradient boosting: supply-temperature MAE 0.067 K, max 0.64 K.
  MLP: MAE 0.077 K, worst error 2.50 K. Gradient boosting was chosen.
* **Held-out test result (gradient boosting, refit on train + validation):**

| Metric on 1231 held-out points in 60 unseen regions | Value |
|---|---|
| Supply temperature, mean absolute error | 0.047 K |
| Supply temperature, 95th percentile error | 0.155 K |
| Supply temperature, worst error | 0.359 K |
| Wet-bulb effectiveness, mean absolute error | 0.0035 |
| Evaporated water, mean absolute error | 0.196 g per kg supply air (worst 1.25) |

Speed: the surrogate predicted all 1231 test points in 0.031 s; one physics call takes about
0.4 s. `tests/test_surrogate.py` re-checks the saved model against 60 fresh physics runs from a
different random seed (bound: MAE < 0.15 K, max < 1.0 K). The surrogate also clamps its answer
at the inlet dew point.

## The assistant

`agent/loop.py` is a manual tool-use loop on the Anthropic Messages API (`claude-sonnet-5`,
key from `ANTHROPIC_API_KEY`). The four tools in `agent/tools.py` are plain Python functions and
are tested directly, with no key.

Guardrails (`agent/guard.py`):

* Every tool result gets an id (`R1`, `R2`, ...). The final answer must cite at least one, and
  every cited id must be a successful result. Otherwise the model gets one retry, then the
  answer is withheld.
* Tools refuse inputs outside the validated envelope instead of extrapolating. If every cooler
  tool call was refused, the model's answer is replaced by a refusal that states the reason,
  whatever the model wrote.
* Bad tool arguments come back to the model as errors, not crashes.

The loop logic is tested with a scripted fake client (`tests/test_agent.py`); the live test
runs only when a key is present.

### A real run on Groq (2026-09-26)

`python -m hvac_cooling_ai ask --provider groq "I'm quoting a job in Phoenix. Will one of these coolers
hold 26 C supply air on a 40 C, 20% RH afternoon? And what about a humid 35 C, 60% RH day?"`

Model `openai/gpt-oss-120b` on Groq, same tools, prompt and citation guard as the Claude loop. The
unedited output is in [`docs/live-transcript-groq-2026-09-26.txt`](docs/live-transcript-groq-2026-09-26.txt).
It answered the dry afternoon from the physics tool (26.48 C supply, cited [R2]) and refused the
humid day because 21 g/kg is outside the validated envelope, instead of guessing.

The first live attempt was withheld by the guard: the model cited results as 【R1】, which the guard
did not recognise. The guard now reads that bracket style too (`test_guard_reads_lenticular_bracket_citations_as_square`).

### Example transcripts (illustrative)

These are **examples, not recorded model runs**: no API key was available while this was built,
so the assistant's wording below was written by hand. **The tool calls and tool results are
real**: they are the outputs of the tools in this repo for these inputs. Reproduce them with
`ToolSession().call(name, args)` from `hvac_cooling_ai.agent.tools`, or the CLI commands above.

> **Q:** Will this cooler hold 26 C supply air on a 44 C, 30% RH afternoon?
>
> `R1 simulate_cooler(dry_bulb_c=44, relative_humidity_pct=30)` -> supply 31.5 C, inlet wet bulb 28.0 C, dew point 22.5 C
> `R2 simulate_cooler(..., channel_velocity_m_s=0.5, working_air_fraction=0.5)` -> supply 26.8 C
>
> **A:** No. At the default design the supply air is about 31.5 C [R1]. Slowing the air to
> 0.5 m/s and sending half of it to the wet side gets it to about 26.8 C [R2], still just above
> your 26 C target. That day is humid for evaporative cooling (dew point 22.5 C [R1]).

> **Q:** How many units for a 12 kW load on a 40 C, 15% RH day, holding the room at 26 C?
>
> `R1 units_needed(sensible_load_kw=12, dry_bulb_c=40, relative_humidity_pct=15, channel_velocity_m_s=0.5, working_air_fraction=0.5)`
> -> supply 18.77 C, 1.71 W of room cooling per channel pair, 7019 channel pairs,
> about 2974 CFM of supply air, about 84 L/h of water
>
> **A:** About 7,019 channel pairs of 200 x 200 mm plates [R1], moving about 2,974 CFM of supply
> air at 18.8 C [R1] and evaporating about 84 L of water an hour [R1]. The modelled core is
> bench-scale, so read this as the size of the heat-exchange surface needed, not a count of
> catalogue units.

> **Q:** What about a 52 C, 10% RH afternoon?
>
> `R1 simulate_cooler(dry_bulb_c=52, relative_humidity_pct=10)` -> refused: inlet dry-bulb 52C is outside the validated range 25-50C
>
> **A (produced by the guard, verbatim):** I can't answer that reliably: the conditions are
> outside the range this cooler model was validated for (inlet dry-bulb 52C is outside the
> validated range 25-50C). [R1] refused the request rather than extrapolate.

## Tests

`pytest -q`: 55 tests (54 run without a key, 1 live-API test skipped). They cover:

* psychrometrics against PsychroLib;
* physics: convergence everywhere in the envelope, outlet below inlet and above the dew point,
  energy balance closing to 1e-6 (product air heat loss vs working air enthalpy gain,
  recomputed from outlet states), every wet channel at or below saturation, a bigger exchanger
  cooling more but never past the dew point, grid sensitivity, linear scaling with stack size;
* surrogate error bound on fresh physics runs, envelope refusal, dew-point clamp;
* tool contracts, bad-input handling, envelope refusal for every cooler tool, CLI;
* the grounding guard and the loop (fake client), including the envelope override;
* the validation numbers and the EER finding.

To check the tests can fail, three defects were planted by hand and each was caught: the dry
side losing 10% less heat than the wet side gains (`test_energy_balance_closes`), the
saturation clamp removed (`test_working_air_gains_moisture`), and the guard accepting
citations of refused results (`test_guard_rejects_missing_unknown_and_failed_citations`).

## Limitations

* **It does not reproduce the source project's numbers** (see Validation). It stays inside the
  dew-point limit, which the published results do not.
* **Geometry, working-air fraction and wet-side arrangement are assumptions.** The source does
  not state them. Different choices change the numbers.
* **Not calibrated against measurements.** The source's model chapter compares its results with
  published literature values; this port has not been compared with any measured data.
* **Bench-scale core.** One channel pair moves 0.46 to 2.8 g/s of air across the envelope. Sizing answers are in
  channel pairs; a real product would need its own geometry and fan data.
* **Fan power is a placeholder** (0.5 W per channel pair, from the source), so COP and EER are
  indicative only.
* **Sea-level pressure only; dew points above 0 C only; laminar flow only** (true across the
  envelope).
* **Mixed wet-side exhaust can read slightly above 100% RH.** Each channel leaves at or below
  saturation; averaging channels at different temperatures produces a little fog. It does not
  affect the supply air.
* **The live assistant was not run for this README.** Its loop and guard are tested with a fake
  client; the live test runs only with a key.

## License

MIT
