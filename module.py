"""Template Syntalos Python module.

This module forwards incoming frames unchanged and emits a dummy float signal
from the tick callback. Copy it into a new repository and replace the dummy
names, ports, metadata, and processing logic with the module-specific behavior.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import final

import numpy as np
import syntalos_mlink as syl


@dataclass
class Settings:
    dummy_period_ms: int = 1_000
    dummy_value: float = 1.0


def serialise_settings(settings: Settings) -> bytes:
    return json.dumps(asdict(settings)).encode()


def deserialise_settings(settings: bytes) -> Settings:
    return Settings(**json.loads(settings.decode()))  # pyright: ignore[reportAny]


@final
class Module:
    def __init__(self, mlink: syl.SyntalosLink) -> None:
        self.mlink = mlink
        self.settings = Settings()
        self.running = False
        self.last_dummy_emit_us: int | None = None

        self.register_ports()
        self.register_callbacks()

    def clear_state(self) -> None:
        self.running = False
        self.last_dummy_emit_us = None

    def on_frame(self, frame: syl.Frame | None) -> None:
        if not self.running or frame is None:
            return

        self.out_frames.submit(frame)

    def submit_dummy_signal(self, timestamp_us: int) -> None:
        block = syl.SignalBlockF32()
        block.timestamps = np.array([timestamp_us], dtype=np.uint64)
        block.data = np.array([[self.settings.dummy_value]], dtype=np.float32)
        self.out_dummy.submit(block)

    # # ################################################################################
    # # Syntalos interface
    # # ################################################################################

    def register_ports(self) -> None:
        self.in_frames = self.mlink.register_input_port(
            "frames-in", "Frames In", syl.DataType.Frame
        )
        self.out_frames = self.mlink.register_output_port(
            "frames-out", "Frames Out", syl.DataType.Frame
        )
        self.out_dummy = self.mlink.register_output_port(
            "dummy-float", "Dummy Float", syl.DataType.SignalBlockF32
        )

    def register_callbacks(self) -> None:
        self.in_frames.on_data = self.on_frame
        self.mlink.on_prepare = self.prepare
        self.mlink.on_start = self.start
        self.mlink.on_stop = self.stop
        self.mlink.on_save_settings = self.save_settings
        self.mlink.on_load_settings = self.load_settings

    def prepare(self) -> bool:
        self.clear_state()

        framerate = self.in_frames.metadata.get("framerate", None)
        if framerate is not None:
            self.out_frames.set_metadata_value("framerate", framerate)

        frame_size = self.in_frames.metadata.get("size", None)
        if frame_size is not None:
            self.out_frames.set_metadata_value_size("size", frame_size)

        self.out_dummy.set_metadata_value("signal_names", ["DUMMY"])
        self.out_dummy.set_metadata_value("time_unit", "microseconds")
        self.out_dummy.set_metadata_value("data_unit", ["a.u."])
        return True

    def start(self) -> None:
        self.running = True

    def event_loop_tick(self) -> None:
        if not self.running:
            return

        now_us = int(syl.time_since_start_usec())
        period_us = max(self.settings.dummy_period_ms, 1) * 1_000
        if self.last_dummy_emit_us is not None and now_us - self.last_dummy_emit_us < period_us:
            return

        self.submit_dummy_signal(now_us)
        self.last_dummy_emit_us = now_us

    def stop(self) -> None:
        self.running = False

    def load_settings(self, settings: bytes, _base_dir: Path) -> bool:
        if not settings:
            return True

        try:
            self.settings = deserialise_settings(settings)
            return True
        except Exception:
            self.settings = Settings()
            raise

    def save_settings(self, _base_dir: Path) -> bytes:
        return serialise_settings(self.settings)


def main() -> int:
    mlink = syl.init_link(rename_process=True)
    _module = Module(mlink)
    mlink.await_data_forever(_module.event_loop_tick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
