Draft for https://github.com/vllm-project/vllm/issues/25700 — not posted.

I'd like to take a scoped migration of `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` to
`--http-timeout-keep-alive` on the Python frontend. The standalone Rust frontend
already exposes that flag, but the Python API/render launchers read the env var
directly, and the Python-to-Rust launcher injects the env value into its JSON
payload.

I checked open PRs referencing #25700, the variable name, and the CLI flag and
did not find a competing migration. The proposed scope is consistent CLI/YAML
configuration across these launch paths, retaining the default of 5 seconds,
preserving zero, and testing parsing/precedence and subprocess forwarding on CPU.
I also found the example API server and EPD proxy consuming the variable and
will account for those users in the migration.

Would a deprecated environment fallback be preferred here, or should this
follow the removal approach used in #55176? My preference is a staged migration
with explicit CLI/YAML values taking precedence over the legacy environment.

I am using AI assistance for code exploration and preparation and will review
the implementation and run the relevant tests before submitting a PR.
