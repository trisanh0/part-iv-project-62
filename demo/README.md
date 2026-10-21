# TEMPO booth demo

One FastAPI process serves one shared room. Benchmark compilation is offline; the live server only reads an immutable, validated catalogue. It does not execute ML benchmarks during a race.

## Run locally

From the repository root, install `requirements.txt`, then set a private presenter code:

```bash
export DEMO_ADMIN_PIN='choose-a-private-code'
.venv/bin/python -m uvicorn demo.server:app --host 127.0.0.1 --port 8000 --workers 1
```

Open `http://localhost:8000`. For phones on the same network, bind to `0.0.0.0` and use the computer's LAN address. Use HTTPS when exposing the booth through the existing Cloudflare tunnel. The Presenter panel first verifies the password, then reveals admin controls (including track selection). The browser retains the verified password only in memory; Lock settings clears it. Every administrative action is still authenticated by the server. There is no default code. Codes must be 4–128 characters.

Presenter authentication is required for starting/resetting a race, changing tracks, adding bots, setting duration, and entering the building phase. The first Ready opens an eight-second entry window. Only ready, connected participants enter the next heat (up to six); people still configuring do not delay it. Additional ready entrants wait for the next heat. A measured baseline bot fills a solo heat. After results, Change your build opens the builder only for the requesting entrant. People reviewing results release their places immediately. Each keeps their own completed result, including through reconnects and subsequent heats, until choosing Change your build. They re-enter setup without being ready automatically. This action requires the completed heat ID and cannot interrupt countdown or racing.

Countdown freezes identities, metrics, catalogue version, duration, start time, and a per-entrant race schedule. Disconnecting or leaving after countdown cannot change the results. An explicit presenter reset cancels the heat. All cars launch together. Classification races use fixed rules in `demo/race.py`:

```text
driving_cost = 1 + 0.12 * ln(1 + compute_seconds)
pit_cost = 4 * (1 - accuracy)
scale = race_duration_ms / max(driving_cost + pit_cost)
```

Every entrant uses the same scale. Finish and pit durations are rounded to milliseconds; driving time is their difference, so the final finish matches the configured deadline exactly. One pit visit around lap one includes travel in driving time and a deterministic accuracy dwell. Equal millisecond finishes share competition positions. Regression and forecasting remain compute-only races. The separate `race_rules_version` does not alter the catalogue scoring version. The weights and compressed animation are simulation rules, not physical benchmark measurements.

A disconnected identity is retained for 30 seconds, with readiness cleared and no place reserved in the next grid. Each tab stores a private resume token in session storage; a reload restores that participant. A replacement connection supersedes the old socket. Launched entrants remain resumable through the race. After the finish, disconnected participants retain the usual 30-second grace period before their places and sessions are released. Server restarts clear room and session state.

## Participant experience

The mobile interface follows Name → Build → Ready → Race → Compare. Fresh visitors remain on the purple landing page during any room phase. Names must contain 1–20 trimmed characters; Start waits for confirmed registration before navigating. Visitors can enter setup during any room phase. Joining by name does not reserve a race place. Ready reserves a place in the next available heat; ready time determines queue order. At the finish, all entrants leave the next grid, bots are removed, and ready waiters enter the next collection window. Back to setup cancels a queued Ready. Disconnected participants immediately lose readiness, while their identity and build remain resumable for 30 seconds. The shared room phase never forces a builder or a result reviewer into somebody else’s race. A valid session resumes its existing participant view after reload.

The default gas-sensor challenge exposes Extract → Select → Predict as three guided stages: NumPy/Numba/tsfresh Minimal, keep-all/ANOVA top-25/subsampled FDR, and logistic regression/random forest. All 18 combinations have eligible recorded measurements. New visitors start with NumPy, keep-all and logistic regression; returning valid selections are preserved. Other measured components remain in the full library. Component changes never silently replace a second stage.

TEMPO is presented as the shared benchmarking framework, with Numba as one extractor plug-in. The builder explains the integration work behind component changes and compares compute, accuracy, extraction RAM and retained feature counts. Compute includes extraction, selection and model fitting; RAM is the peak during extraction. Selection and readiness follow server acknowledgements. Confirmed changes show measurement deltas, and session-local history compares the previous completed build only when participant, challenge and catalogue build match. Race times are not compared across heats because normalisation depends on the entrants. The replay does not claim a new benchmark run, significance from a single measurement, or reproduction of the poster's aggregate figures.

Results show the actual finish, driving/pit breakdown, pipeline and measured time/quality/RAM. Technical details and the benchmark scatter plot are optional disclosures. Presenter controls remain separately authenticated.

The application uses framework-free HTML/CSS/modules with shared presentation helpers in `demo/static/js/presentation.js`. The stylesheet is a single design system; responsive checks cover 320, 390, and 430px widths. Zoom is enabled, fields use at least 16px text, and action controls use at least 48px touch targets.

## Compile and publish data

```bash
.venv/bin/python -m demo.compile_db
```

Run compilation from the repository root. The compiler creates:

- `demo/data/builds/<build-id>/demo_benchmarks.db`: read-only runtime tables plus original measurement payloads and row-level provenance.
- `demo/data/builds/<build-id>/demo_catalog.json`: full-precision runtime configurations and track metadata.
- `demo/data/builds/<build-id>/TM…__demo_catalogue_audit.json`: source/configuration hashes, cohort selection, donor rows, and exclusion reasons.
- `demo/data/current.json`: the atomically replaced release pointer and artifact checksums.

The previous release stays usable if compilation or validation fails. Existing servers remain pinned to their startup release; restart after publishing. Do not edit published build files. The original flat `demo_benchmarks.db` and `demo_catalog.json` are legacy v1 artifacts and are not loaded by the v2 server.

Every source CSV must map unambiguously to an experiment configuration via `output_dir`. Run-local configuration snapshots take precedence. Unknown settings, conflicting duplicate seeds, inconsistent feature counts, unsupported schemas, missing scores, and incomplete pipeline costs are excluded. Identical duplicate seeds are counted once.

One source cohort is selected per track: most eligible configurations, then most directly measured configurations, then source path. Separate experiments are never averaged together. Cached rows may reuse a measured extraction time/RAM pair only within the same source file, dataset, seed, extractor settings, and feature count. Original missing measurements remain missing in raw storage; derived values identify their donor row. There are no cross-dataset or arbitrary numeric fallbacks. Large forecast trajectory arrays remain in their source CSV; the database preserves their source row IDs and source hashes alongside original measurement columns.

Tracks need at least two eligible configurations. A `tsfresh_efficient` baseline is preferred, followed by `tsfresh_minimal`. Without either, relative speedup and memory reduction are null; the slowest valid configuration serves as the comparison bot.

The current filtered release contains **288 configurations across 10 tracks**, from 3,494 input rows. Appliances-energy forecasting and Beijing forecasting lack eligible physical measurements. The unversioned `drift-bifurcation-reg` and `pred-maintenance` tracks lack registered track metadata and are excluded. BEED and HAR currently omit Numba because the selected source has ambiguous Numba parameter variants. The audit report records all exclusions. Coverage can be expanded by supplying an unambiguous, measured experiment and recompiling; this change does not rerun benchmarks.

Historical files do not always contain dataset-content hashes or clean source revisions. A recorded source/configuration hash identifies the available evidence; it does not retroactively certify missing historical provenance.

## Scoring and query interface

Catalogue composite rankings use `demo.scoring.composite_score`:

```text
100*q - 15*log10(max(total_latency_seconds, 0.001)) - 5*extraction_ram_mb/1000
q = clip(accuracy, 0, 1)                for classification
q = clip((R² + 1)/2, 0, 1)              for regression
q = 1/(1 + RMSE/100)                    for forecasting
```

Raw scores remain unchanged. Lower RMSE is better; higher accuracy/R² is better. Pareto membership uses the raw metric direction, latency, and RAM. Ties share awards; IDs provide deterministic display ordering. Calculations retain full precision.

```python
from demo.query import DemoQueryEngine

engine = DemoQueryEngine()  # resolve and validate current release once
tracks = engine.list_tracks(flagship_only=True)
choices = engine.get_valid_options("beed")["configurations"]
metrics = engine.get_config("beed", **choices[0])
leaders = engine.get_leaderboard("beed", sort_by="pareto", limit=10)
rows = engine.execute_custom_query("SELECT COUNT(*) AS n FROM raw_benchmark_runs")
```

The query engine loads a validated in-memory snapshot of both artifacts. `prefer_memory_cache` remains accepted for call compatibility; all runtime lookups use that snapshot. SQL inspection uses a separate read-only connection and authorizer. v1 schema artifacts are deliberately rejected.

## HTTP and WebSocket contracts

Read routes remain `/api/state`, `/api/tracks`, `/api/options/{track}`, `/api/pareto/{track}`, and `/api/leaderboard/{track}`. Options now include actual `configurations`, not merely independent component lists. Unknown tracks return 404; invalid query values return 422. `/api/admin/action` accepts JSON `{action, pin, ...parameters}` and uses the same dispatcher as WebSockets.

Connect to `/ws/game` and send this first frame within five seconds:

```json
{"action":"hello","protocol_version":2,"resume_token":null}
```

`INIT.data` returns a private `resume_token`, `client_id`, and state. Subsequent frames use `join`, `leave`, `set_pipeline`, `set_ready`, `race_again` (with `heat_id`), or `ping`; presenter commands also require `pin`. Invalid messages produce an `ERROR` event without mutating state. Obsolete protocol versions receive `reload_required` and close.

Events include protocol version, state revision, heat ID, and server time. State additionally includes the catalogue build ID, scoring version, race rules version, phase-start time, authoritative race-start time, and the ready-window deadline. INIT and STATE_UPDATE are personalised with `participant.view` and `personal_results`; `/api/state` omits that personal context. The shared `participants` list excludes disconnected people and result reviewers. Frozen active racers carry `race_schedule` with driving, pit entry/stop/departure/exit, and finish times in relative milliseconds; result rows retain that schedule and include `finish_position` and `tied`. The RACE_START event includes the same scheduled racers. Metrics expose selection time, extracted/selected feature counts and sample count. Tokens never appear in broadcast state. Reconnecting clients use server timing rather than starting playback again.

Each socket has a 32-message outbound queue and a two-second send deadline. Overflow or timeout closes that connection without blocking other clients. Reset and shutdown cancel and await background tasks. Logs report transitions, rejected commands, and connection failures without credentials or message payloads.

## Deployment and checks

```bash
docker compose -f docker-compose.demo.yml build
docker compose -f docker-compose.demo.yml up -d
# Optional public tunnel; configure its origin to http://tempo-demo:8000:
CLOUDFLARE_TUNNEL_TOKEN='your-token' docker compose -f docker-compose.demo.yml --profile tunnel up -d

.venv/bin/python -m pytest tests/test_demo_game_state.py tests/test_demo_query.py tests/test_demo_server.py tests/test_demo_compiler.py -q
```

Set `DEMO_ADMIN_PIN` in the deployment environment before invoking Compose. The image fixes Uvicorn to one worker. Multiple workers/replicas would create independent rooms and are unsupported. Other settings are `DEMO_DEFAULT_TRACK` (default `gas-sensor-drift-sub`), `DEMO_MAX_ACTIVE_RACERS` (2–6, default 6), `DEMO_RACE_DURATION_SECONDS` (5–60), and `DEMO_COUNTDOWN_SECONDS` (0–30).

The health check reads `/api/state`; startup validates catalogue versions, matching JSON/SQLite contents, and checksums. Test coverage includes authorization on both transports, immutable results, reconnection, slow clients, invalid combinations, score direction, cohort isolation, and failed publication. Docker build execution requires a Docker installation; it was unavailable on the implementation workstation.

Client regression tests (Node.js):

```bash
node --test tests/test_demo_client.mjs
```

The live leaderboard sits below the circuit and shows position, progress, pit state and simulated finish time. Deterministic variation stays between fixed checkpoints. Client animation, reconnect playback and results all use the server schedule; composite scores and awards remain in the API for compatibility without deciding the displayed winner.

The presenter picker offers gas sensors, industrial classification and BEED by default. Show all tracks reveals the full eligible library; regression and forecasting are labelled compute-only. An active track outside the curated list remains visible until changed. Track choice stays fixed across rounds and duplicate builds remain allowed.

The race view combines an original Grand Prix-style circuit with open-wheel car silhouettes, a two-column starting grid, a pit lane and garage block, a YOU marker, and an elapsed/position/lap strip. The long main straight, broad sweepers and linked infield turns are drawn as one continuous spline. Paper and restrained purple details match the site. Starting-grid offsets converge onto the race line after launch; they do not change authoritative progress or finish times. All cars move immediately, and tied finishers share their final position. A periodic geometry-based pace profile uses broad, filtered braking zones and continuous velocity interpolation, avoiding stop/start motion in linked bends. Continuous position and tangent interpolation prevents heading snaps; lane offsets narrow smoothly at tight apexes avoiding the near-stops caused by a fixed inside offset. A subtle deterministic pace wave is confined between pit and finish checkpoints. Pit-entry braking matches the pit-lane velocity; departure accelerates smoothly back onto the circuit. Six independent pit boxes allow simultaneous service without an invented queue. Inverting accumulated lap time into distance preserves monotonic progress, smooth lap boundaries, authoritative completion times and reconnect recovery. This is illustrative pacing rather than a full vehicle physics simulation.

On touch devices, focusing the landing name field adds scroll space immediately and keeps the field inside the visible viewport as the keyboard opens. The default race lasts 27 seconds, preceded by a seven-second automatic start. The gantry fades in dark, five paired red lights illuminate one second apart, then all lights go out after a short shared hold. Reconnecting clients resume the current stage; reduced-motion preferences suppress fades without altering timing. The start line, six-car grid and pit garages sit along the middle of the main straight.

“Explore benchmark results” is optional results detail. Its line is the two-dimensional latency/quality frontier, recomputed from the measured configurations; the backend’s three-objective frontier remains available through the API. Better quality is always higher on the chart (including lower forecast RMSE). A labelled selector lets touch and keyboard users inspect every build, including racers sharing the same coordinates. Memory appears in the selected build’s measurements, outside the plotted trade-off.

Frontend files are snapshotted together when the server module loads. HTML is served with `Cache-Control: no-store` and references a content-versioned `/assets/<version>/` path; relative JavaScript imports stay under that same version. This prevents cached track geometry from mixing with a newer renderer. Restart the server after editing frontend files, then refresh clients. Legacy `/static/` paths remain available with caching disabled.

Pipeline stages and optional details share one native-details animation, including keyboard activation, rapid reversal and reduced-motion support. Other driver names are omitted from the circuit; the participant’s own car is labelled YOU. Cars continue beyond the line and decelerate without changing classified progress or finish time. A finish notice announces the participant’s position (including shared wins), and the race view remains for 2.5 seconds after the last finish before opening personal results. Peak extraction RAM is shown alongside time and prediction quality; it does not contribute an invented physical weight penalty.
