# First vLLM contribution: HTTP keep-alive configuration

## Learning notes

- [เรียนรู้ก่อนทำ llm-d #1525: P/D disaggregation และ request draining](llm-d-request-draining-study.md)
  — พื้นฐาน tensor, KV cache, parallelism, Kubernetes shutdown และแผนอ่านโค้ดพร้อมคำถามทบทวน

The original contribution plan below is a historical preparation note; its status
statements describe the planning stage, not the current PR status.

Selected task under [vLLM #25700](https://github.com/vllm-project/vllm/issues/25700):
expose HTTP idle keep-alive timeout through the Python frontend CLI and migrate
its environment-variable configuration consistently across the serving paths.

Status: selection and code-path investigation complete; implementation and
runtime tests have not started. No GitHub comment or PR has been posted.

## Why this task

The Python API and render servers read `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` directly.
The standalone Rust frontend already exposes `--http-timeout-keep-alive`.
Python's Rust-process launcher explicitly forwards the environment value in its
JSON payload, so merely adding a Python CLI field would still leave an override
bug in that path.

The contribution should make this serving option discoverable in Python's CLI
and YAML configuration and apply the resolved value consistently. This teaches
argument parsing, configuration precedence, server launch, and subprocess
configuration without requiring model inference or GPU kernels.

HTTP keep-alive here means the time an **idle connection between requests** may
remain open. It is not the maximum generation time, an engine startup timeout,
or the interval for SSE heartbeat comments during streaming.

## Checkout and tools

- Independent upstream checkout: `vllm/` (ignored by this parent repository).
- Inspected upstream commit: `54d93af9fb89d0bb8392bdb72db42bb6a17cb4ca`.
- Inspection date: 2026-10-05.
- Python 3.12 virtual environment: `vllm/.venv/`.
- Installed `requirements/lint.txt` and pre-commit hooks.
- Full vLLM/test dependencies are not installed. Hooks have not been run.

Follow `vllm/AGENTS.md`: use `uv` and `.venv/bin/python` for Python work.
On this workspace use `UV_CACHE_DIR=/tmp/vllm-uv-cache` and
`PRE_COMMIT_HOME=/tmp/vllm-pre-commit` to keep tool caches writable.

## Code-reading map

Read these in order inside `vllm/`:

| File | What to follow |
| --- | --- |
| `vllm/envs.py` | Declaration, integer parser, default of 5 seconds, exclusion from compilation hash |
| `vllm/entrypoints/launchers/cli_args.py` | `FrontendArgs`, generated CLI options, `validate_parsed_serve_args` |
| `vllm/entrypoints/launchers/api_server/entry.py` | `build_and_serve` passes the timeout to Uvicorn |
| `vllm/entrypoints/launchers/render/entry.py` | Equivalent path for the CPU-only rendering server |
| `vllm/v1/utils.py` | `RustFrontendProcessManager` constructs the `--args-json` payload |
| `rust/src/cmd/src/cli.rs` | Existing Rust CLI flag, environment fallback, and timeout resolution |
| `examples/applications/api_server/server.py` | Separate example server also reads the environment variable |
| `examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py` | Client pool timeout derives from the same environment variable |

Useful search:

```bash
cd vllm
rg -n 'VLLM_HTTP_TIMEOUT_KEEP_ALIVE|http_timeout_keep_alive|timeout_keep_alive' \
  vllm rust examples tests docs
```

## Proposed behavior and scope

The intended user-facing result is:

```bash
vllm serve <model> --http-timeout-keep-alive 30
```

Or in a serving YAML config:

```yaml
http-timeout-keep-alive: 30
```

Keep the existing default of 5 seconds. Preserve zero as disabling idle
connection reuse; reject negative values before launching the server.
The CLI should override YAML consistently with other serving arguments.

Before writing the patch, settle the environment compatibility policy. My
recommended initial design is a deprecated fallback, with precedence
`explicit CLI/YAML > legacy environment variable > 5`. This avoids silently
changing existing deployments; it is a staged migration, not immediate removal
of the variable. Check recent accepted migrations and maintainer preference
before finalizing that policy. The related merged migration #55176 used removal,
so a hard removal is also a precedent, not an automatic requirement here.

Do not use `value or default`: zero is meaningful. Do not read a dynamic
environment fallback into an import-time constant. The Rust launcher's existing
environment assignment must not overwrite an explicit CLI/YAML value.

The example proxy requires special care: it retires pooled connections before
the backend's idle timeout, and uses `force_close` when that timeout is zero.
If removing the environment variable, give the proxy an explicit way to receive
the backend timeout and update its tests/documentation. Do not silently leave
it assuming 5 seconds when the backend is configured differently.

## CPU validation plan

Use the existing test suites, with mocked engine/process boundaries where they
already provide them. No model weights or cloud GPU should be needed for the
core regression tests, although importing vLLM tests still requires its Python
dependencies and a supported development installation.

1. Extend `tests/entrypoints/launchers/test_cli_args.py`: defaults, zero,
   positive values, negative rejection, YAML, CLI-over-YAML precedence, and
   legacy-environment precedence if retaining compatibility.
2. Extend `tests/entrypoints/launchers/api_server/test_api_server_process_manager.py`:
   inspect the actual Rust subprocess JSON and ensure an explicit setting is
   not replaced by the old environment value.
3. Verify the API and renderer pass the configured value to the HTTP server,
   without starting an inference engine. Prefer existing launcher fixtures.
4. If modifying proxy configuration, extend
   `tests/v1/ec_connector/unit/test_epd_proxy_retry.py` to preserve connection
   retirement and zero-timeout behavior.
5. Run a small local HTTP smoke check for idle connection behavior if needed;
   GPU inference is unrelated to that contract.

Test design: the contract is that the user's selected timeout reaches the
server unchanged. The main failure to prevent is an override being ignored in
one launch path, especially Python-to-Rust serialization. Check that contract
at the argument/launcher boundary rather than testing model output.

## Duplicate-work checks

Read #25700 and its comments and searched GitHub for open PRs with:

- `25700 in:body`
- `VLLM_HTTP_TIMEOUT_KEEP_ALIVE`
- `"http-timeout-keep-alive"` (including closed PRs for context)
- `"keep alive"` and `"keep-alive" "config"`

No direct competing migration was found in those results at inspection time.
This does not establish ownership or maintainer acceptance. Recheck before
starting the patch and before submission.

`gh` is not installed here, so equivalent GitHub REST issue/comment/search
endpoints were used. Relevant adjacent work:

- [#18472](https://github.com/vllm-project/vllm/pull/18472): original environment-based configuration.
- [#45890](https://github.com/vllm-project/vllm/pull/45890): Rust frontend HTTPS/mTLS work.
- [#47840](https://github.com/vllm-project/vllm/pull/47840): open Rust HTTP body-limit work; different setting.
- [#55176](https://github.com/vllm-project/vllm/pull/55176): scale-out endpoint environment-to-CLI migration.

The Responses API store variable is already claimed in #25700; it was excluded.

## Before submitting

The human submitter must understand and review every changed line and run the
relevant tests, as required by upstream `AGENTS.md`. Follow upstream's
`.agents/skills/pr-checklist/SKILL.md` before submission and disclose AI
assistance. This preparation is not a submitted or merge-ready contribution.
