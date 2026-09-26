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

Around that core: `surrogate/drift.py` (surrogate vs physics and input drift checks),
`agent/usage.py` (latency, tokens and estimated cost per LLM call), `agent/eval_cases.json` +
`agent/evals.py` (prompt and workflow regression suite), and `lambda_handler.py` + `infra/`
(Terraform for AWS Lambda, validated, not deployed).

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m hvac_cooling_ai simulate --t 44 --rh 30          # physics model
python -m hvac_cooling_ai predict  --t 44 --rh 30          # surrogate (milliseconds)
python -m hvac_cooling_ai size --t 40 --rh 15 --load-kw 12 --velocity 0.5 --r 0.5
python -m hvac_cooling_ai psychro  --t 44 --rh 30
python -m hvac_cooling_ai validate                         # comparison with the source project
ANTHROPIC_API_KEY=... python -m hvac_cooling_ai ask "Will this cooler hold 26 C supply air on a 44 C, 30% RH afternoon?" --show-tools --show-usage

python -m hvac_cooling_ai drift --n 100                    # surrogate vs physics on 100 fresh points
python -m hvac_cooling_ai drift --inputs examples/requests-heatwave.csv   # input drift for a request log

pytest -q          # 104 pass; 16 live-API tests are skipped without a key
ruff check . && ruff format --check .
python -m hvac_cooling_ai.surrogate.train                  # regenerate the surrogate (about 1 minute)
infra/build_lambda.sh && terraform -chdir=infra init -backend=false && terraform -chdir=infra validate
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

## Drift detection

`surrogate/drift.py`, CLI `drift`. Two questions: is the saved surrogate still faithful to the
physics, and does real traffic look like what it was trained on?

**Surrogate vs physics.** Sample fresh points inside the envelope on a seed never used in training,
run both models, and compare. The limits are the held-out test errors recorded in `metrics.json`
times 3, so the bar follows the model that shipped. This catches a stale artifact after the physics
changes, a swapped or corrupted model file, or a library upgrade that shifts predictions. Exit code 4
means drift. CI runs it on 100 points on every push.

`python -m hvac_cooling_ai drift --n 100` (run 2026-09-26):

| Supply temperature error | Observed, 100 fresh points | Limit (3 x recorded test error) |
|---|---|---|
| Mean absolute error | 0.033 K | 0.142 K |
| 95th percentile | 0.122 K | 0.464 K |
| Worst | 0.168 K | 1.077 K |
| Evaporated water, mean absolute error | 0.149 g/kg | 0.588 g/kg |

Result: passed. Planted drift trips it (`tests/test_drift.py`, 40 points like CI): the saved model
with 0.01 added to its wet-bulb effectiveness (about 0.1 K) measured MAE 0.149 K against the
0.142 K limit and failed; with 0.02 it measured 0.280 K and failed; the unmodified model measured
0.050 K on the same points and passed. Physics that drifted 3% under a stale surrogate also fails.

**Input drift.** Give it a CSV or JSON of real requests (the same field names the tools take:
`dry_bulb_c`, `relative_humidity_pct` or `humidity_ratio_g_kg`, and optionally
`channel_velocity_m_s`, `working_air_fraction`). It reports the share outside the training envelope
(the tools refuse those) and a population stability index per feature against the exact 6000
training inputs, regenerated from the recorded seed. It fails above 5% outside the envelope or a PSI
above 0.25. Features a request log does not supply are not scored, because the tool default is a
design choice, not traffic. Fewer than 50 valid rows get no PSI.

The two files in `examples/` are **synthetic** (written by `examples/make_requests.py`), not real
traffic:

| `python -m hvac_cooling_ai drift --inputs ...` | Outside envelope | PSI dry-bulb | PSI humidity | Result |
|---|---|---|---|---|
| `examples/requests-training-like.csv` (500 rows) | 0 | 0.033 | 0.024 | pass |
| `examples/requests-heatwave.csv` (same rows, 8 C hotter) | 154 (30.8%) | 2.491 | 0.024 | fail |

The training sample is uniform over the envelope, so real traffic that clusters (all dry, all hot)
will show a high PSI even inside the envelope. That is a signal to look at accuracy in that region,
not proof the answers are wrong; the surrogate-vs-physics check is what measures accuracy.

## Cost and latency tracking

`agent/usage.py`. Both loops record every LLM request on `AgentResult.llm_calls`: wall-clock
latency, input and output tokens from the API's own usage fields (Anthropic `input_tokens`,
`output_tokens` and the cache fields; Groq/OpenAI `prompt_tokens`, `completion_tokens`) and an
estimated cost. `AgentResult.usage` sums them; `ask --show-usage` prints them.

Prices are **configurable estimates** in code, US dollars per million tokens, read 2026-09-26:
`claude-sonnet-5` $2.00 in / $10.00 out (Anthropic list price; cache writes priced at 1.25x input,
cache reads at 0.1x) and `openai/gpt-oss-120b` $0.15 in / $0.60 out (Groq's model page). Point
`HVAC_AI_PRICES` at a JSON file to change them. A model with no price, or a response with no usage
block, shows `n/a` rather than a guessed figure. Groq latency includes any rate-limit wait.

Output of a **scripted fake run** (no network, so latency is 0.00 s; the token counts are the
script's, not a real model's), the same path `tests/test_usage.py` checks:

```
call 1  openai/gpt-oss-120b  0.00 s  in 1200 tok  out 80 tok  $0.00023
call 2  openai/gpt-oss-120b  0.00 s  in 1200 tok  out 80 tok  $0.00023
total  2 calls  0.00 s  in 2400 tok  out 160 tok  est. $0.00046 (prices are estimates as of 2026-09-26)
```

## Prompt and workflow regression suite

`agent/eval_cases.json` holds 9 contractor questions. Each states the expected outcome
(`grounded`, `refused` outside the envelope, or `withheld` by the citation guard), the tools that
must be called, and argument checks (for example, 104 F must reach the tool as 40 C, within 0.5).
`tests/test_prompt_regression.py` replays each case through scripted Anthropic and Groq clients, so
all 9 run on both loops in CI (18 tests) with tools, guard and usage tracking in the path. With
`ANTHROPIC_API_KEY` or `GROQ_API_KEY` set, the 7 cases marked live send the same questions to the
real model and apply the same expectations; without a key those 14 tests are skipped.

Controls: a guard planted to accept every answer turns the withheld and both refused cases red
(`test_suite_catches_a_disabled_guard`), and a model that passes 104 as Celsius is reported both for
the wrong outcome and for the wrong argument. The suite also caught a real bug while this was being
built: the Groq loop's usage list was overwritten by a variable of the same name, and all 9 Groq
cases failed until it was fixed.

## Deploying the tools on AWS (Terraform): validated, not deployed

**This has been validated, not deployed.** There are no AWS credentials in this project, so
`terraform plan` and `apply` have never run against an account.

`infra/` would put the four tools (not the LLM, so no API keys in the cloud) behind one Lambda
function. `src/hvac_cooling_ai/lambda_handler.py` takes function-URL events: `GET` lists the tools
and the envelope; `POST {"tool": ..., "args": {...}}` returns 200, 422 when refused outside the
envelope, or 400/405/413 for bad requests.

* **Function:** Python 3.12 on arm64, 1024 MB, 30 s timeout, reserved concurrency 5 so a runaway
  caller cannot run up the bill.
* **Access:** a function URL with `AWS_IAM` auth, never public. `invoker_principal_arns` grants
  named principals both permissions AWS now requires (`lambda:InvokeFunctionUrl`, and
  `lambda:InvokeFunction` only when called through the URL); the `invoker_policy_json` output is
  the matching identity policy for same-account callers.
* **Least privilege:** the execution role can only write log streams in its own log group, which
  Terraform creates with 14-day retention. It has no S3 access; Lambda fetches the code with the
  deployer's credentials.
* **Package:** `infra/build_lambda.sh` installs Linux arm64 wheels. The zip is 61,170,195 bytes
  (over Lambda's 50 MB direct upload, so it goes through a private, encrypted, versioned S3 bucket
  under a content-addressed key) and 220,834,383 bytes unzipped, under the 262,144,000 byte limit;
  the script fails if that stops being true.

Checks run: `terraform fmt -check -recursive`, `terraform init -backend=false` and
`terraform validate` pass locally with Terraform 1.16.4 and AWS provider 6.66.0, and in CI; a
misspelt argument planted in `main.tf` made `validate` fail. `tests/test_lambda_handler.py` covers
the handler, and checks that the handler string in `main.tf` imports this function. The built zip
was also run in AWS's public `python:3.12` arm64 Lambda base image with its local runtime emulator
(Docker, 2026-09-26): list tools 200, physics 200 in 439 ms, surrogate 200 in 1708 ms (first call
loads the model), refusal 422 in 0.66 ms. The emulator does not enforce the memory size, so those
are not Lambda timings.

## Tests

`pytest -q` (2026-09-26): 104 passed, 16 skipped. The 16 skipped are the live-API tests (2 older
ones plus the 7 live regression cases on each of Claude and Groq), which run only with a key. They
cover:

* psychrometrics against PsychroLib;
* physics: convergence everywhere in the envelope, outlet below inlet and above the dew point,
  energy balance closing to 1e-6 (product air heat loss vs working air enthalpy gain,
  recomputed from outlet states), every wet channel at or below saturation, a bigger exchanger
  cooling more but never past the dew point, grid sensitivity, linear scaling with stack size;
* surrogate error bound on fresh physics runs, envelope refusal, dew-point clamp;
* tool contracts, bad-input handling, envelope refusal for every cooler tool, CLI;
* the grounding guard and the loop (fake client), including the envelope override;
* the validation numbers and the EER finding;
* drift checks with planted drift (biased model, changed physics, hotter and drier traffic);
* token, latency and cost parsing for both API formats, price overrides, `ask --show-usage`;
* the prompt and workflow regression suite on both loops;
* the Lambda handler, and that the Terraform handler string points at it.

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
* **The live regression cases and live usage figures have not been run.** No key was available
  when they were added; the only real model run is the Groq transcript above, recorded before
  usage tracking existed.
* **Prices are estimates** as of 2026-09-26 and will go stale; override them with `HVAC_AI_PRICES`.
* **The AWS stack is validated, not deployed.** No `plan` or `apply` has run against an account.

## License

MIT
