# Syntalos Python Module Template

Template for a Syntalos Python module.

The default module:

- forwards a `Frame` input to a `Frame` output through an `on_data` callback
- emits a dummy `SignalBlockF32` output from the tick callback
- persists a `Settings` dataclass through Syntalos save/load callbacks
- uses a per-module virtual environment with `requirements.txt`

When creating a new module, update `module.toml`, replace the dummy names, and
adjust the ports, metadata, settings, and processing logic in `module.py`.
