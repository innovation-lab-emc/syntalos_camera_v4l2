"""Syntalos module for UVC/V4L2 cameras via pyrav4l2."""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
import queue
import sys
import threading
import time
import traceback
from typing import Any, Callable, final

import cv2 as cv
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from pyrav4l2 import Device, FrameInterval, FrameSize, Stream, StreamFrameMetadata
from pyrav4l2.controls import Control, IntegerMenuItem, Item, Menu, MenuItem
from pyrav4l2.device import ColorFormat
from pyrav4l2.v4l2 import (
    V4L2_BUF_FLAG_TIMESTAMP_COPY,
    V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC,
    V4L2_BUF_FLAG_TIMESTAMP_UNKNOWN,
    V4L2_BUF_FLAG_TSTAMP_SRC_EOF,
    V4L2_BUF_FLAG_TSTAMP_SRC_SOE,
    V4L2_CTRL_TYPE_BITMASK,
    V4L2_CTRL_TYPE_BOOLEAN,
    V4L2_CTRL_TYPE_BUTTON,
    V4L2_CTRL_TYPE_CTRL_CLASS,
    V4L2_CTRL_TYPE_INTEGER,
    V4L2_CTRL_TYPE_INTEGER64,
    V4L2_CTRL_TYPE_STRING,
    V4L2_PIX_FMT_BGR24,
    V4L2_PIX_FMT_GREY,
    V4L2_PIX_FMT_JPEG,
    V4L2_PIX_FMT_MJPEG,
    V4L2_PIX_FMT_RGB24,
    V4L2_PIX_FMT_UYVY,
    V4L2_PIX_FMT_Y16,
    V4L2_PIX_FMT_YUYV,
)
import syntalos_mlink as syl

import logging


L = logging.getLogger("camera-v4l2")

FRAME_QUEUE_MAX = 16
DECODE_WARNING_INTERVAL = 30
DROP_WARNING_INTERVAL = 30
SPINBOX_MIN = -(2**31)
SPINBOX_MAX = 2**31 - 1
V4L2_SEQUENCE_MODULUS = 2**32

V4L2_CID_BASE = 0x00980900
V4L2_CID_AUTO_WHITE_BALANCE = V4L2_CID_BASE + 12
V4L2_CID_WHITE_BALANCE_TEMPERATURE = V4L2_CID_BASE + 26

V4L2_CID_CAMERA_CLASS_BASE = 0x009A0900
V4L2_CID_EXPOSURE_AUTO = V4L2_CID_CAMERA_CLASS_BASE + 1
V4L2_CID_EXPOSURE_ABSOLUTE = V4L2_CID_CAMERA_CLASS_BASE + 2
V4L2_CID_EXPOSURE_AUTO_PRIORITY = V4L2_CID_CAMERA_CLASS_BASE + 3
V4L2_CID_FOCUS_ABSOLUTE = V4L2_CID_CAMERA_CLASS_BASE + 10
V4L2_CID_FOCUS_AUTO = V4L2_CID_CAMERA_CLASS_BASE + 12
V4L2_EXPOSURE_MANUAL = 1

AUTO_CONTROL_DEPENDENT_IDS = {
    V4L2_CID_AUTO_WHITE_BALANCE: V4L2_CID_WHITE_BALANCE_TEMPERATURE,
    V4L2_CID_EXPOSURE_AUTO: V4L2_CID_EXPOSURE_ABSOLUTE,
    V4L2_CID_FOCUS_AUTO: V4L2_CID_FOCUS_ABSOLUTE,
}

AUTO_CONTROL_IDS = {
    V4L2_CID_AUTO_WHITE_BALANCE,
    V4L2_CID_EXPOSURE_AUTO,
    V4L2_CID_FOCUS_AUTO,
}

DEPENDENT_CONTROL_IDS = {
    V4L2_CID_WHITE_BALANCE_TEMPERATURE,
    V4L2_CID_EXPOSURE_ABSOLUTE,
    V4L2_CID_FOCUS_ABSOLUTE,
}

CONTROL_NAME_PRIORITIES = {
    "auto_white_balance": 0,
    "white_balance_temperature_auto": 0,
    "exposure_auto": 0,
    "auto_exposure": 0,
    "focus_auto": 0,
    "autofocus": 0,
    "exposure_auto_priority": 1,
    "white_balance_temperature": 20,
    "exposure_absolute": 20,
    "focus_absolute": 20,
}

JsonControlValue = bool | int | str
ControlValueSetter = Callable[[JsonControlValue], None]

SUPPORTED_PIXEL_FORMATS = {
    V4L2_PIX_FMT_MJPEG,
    V4L2_PIX_FMT_JPEG,
    V4L2_PIX_FMT_YUYV,
    V4L2_PIX_FMT_UYVY,
    V4L2_PIX_FMT_RGB24,
    V4L2_PIX_FMT_BGR24,
    V4L2_PIX_FMT_GREY,
    V4L2_PIX_FMT_Y16,
}

VALUE_CONTROL_TYPES = {
    V4L2_CTRL_TYPE_BITMASK,
    V4L2_CTRL_TYPE_BOOLEAN,
    V4L2_CTRL_TYPE_INTEGER,
    V4L2_CTRL_TYPE_INTEGER64,
    V4L2_CTRL_TYPE_STRING,
}


@dataclass
class Settings:
    device_path: str = ""
    pixel_format: int = 0
    frame_width: int = 0
    frame_height: int = 0
    interval_numerator: int = 0
    interval_denominator: int = 0
    control_values: dict[str, JsonControlValue] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoDeviceInfo:
    path: str
    name: str
    driver: str


@dataclass(frozen=True)
class CaptureConfig:
    device_path: str
    pixel_format: int
    frame_width: int
    frame_height: int
    interval_numerator: int
    interval_denominator: int
    control_values: dict[str, JsonControlValue]
    start_monotonic_ns: int = 0
    start_syl_us: int = 0


@dataclass(frozen=True)
class ControlUpdate:
    control_id: int
    value: JsonControlValue


class ControlUpdateBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._updates: dict[int, ControlUpdate] = {}

    def put(self, update: ControlUpdate) -> None:
        with self._lock:
            self._updates[update.control_id] = update

    def drain(self) -> list[ControlUpdate]:
        with self._lock:
            updates = list(self._updates.values())
            self._updates.clear()
        return updates


@dataclass(frozen=True)
class CapturedFrame:
    index: int
    time_usec: int
    mat: np.ndarray


@dataclass(frozen=True)
class CaptureError:
    message: str


class FrameDecodeError(ValueError):
    pass


def monotonic_clock_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def syntalos_clock_anchor() -> tuple[int, int]:
    monotonic_before_ns = monotonic_clock_ns()
    start_syl_us = int(syl.time_since_start_usec())
    monotonic_after_ns = monotonic_clock_ns()
    return (monotonic_before_ns + monotonic_after_ns) // 2, start_syl_us


def v4l2_timestamp_type_name(timestamp_type: int) -> str:
    if timestamp_type == V4L2_BUF_FLAG_TIMESTAMP_UNKNOWN:
        return "unknown"
    if timestamp_type == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC:
        return "monotonic"
    if timestamp_type == V4L2_BUF_FLAG_TIMESTAMP_COPY:
        return "copy"
    return f"0x{timestamp_type:08x}"


def v4l2_timestamp_source_name(timestamp_source: int) -> str:
    if timestamp_source == V4L2_BUF_FLAG_TSTAMP_SRC_EOF:
        return "end-of-frame"
    if timestamp_source == V4L2_BUF_FLAG_TSTAMP_SRC_SOE:
        return "start-of-exposure"
    return f"0x{timestamp_source:08x}"


def v4l2_sequence_gap(previous_sequence: int, current_sequence: int) -> int | None:
    delta = (current_sequence - previous_sequence) % V4L2_SEQUENCE_MODULUS
    if delta == 1:
        return 0
    if 1 < delta < (V4L2_SEQUENCE_MODULUS // 2):
        return delta - 1
    return None


def v4l2_timestamp_to_syntalos_us(metadata: StreamFrameMetadata, config: CaptureConfig) -> int:
    if metadata.timestamp_type != V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC:
        raise TypeError(f"{metadata.timestamp_type=} != {V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC = }")
    # V4L2 reports CLOCK_MONOTONIC time; Syntalos Frame.time is time since run start.
    return max(
        0,
        config.start_syl_us + (metadata.timestamp_us - config.start_monotonic_ns // 1_000),
    )


def serialise_settings(settings: Settings) -> bytes:
    return json.dumps(asdict(settings)).encode()


def deserialise_settings(settings: bytes) -> Settings:
    if not settings:
        return Settings()

    raw: dict[str, Any] = json.loads(settings.decode())
    control_values = raw.get("control_values", {})
    if not isinstance(control_values, dict):
        control_values = {}

    return Settings(
        device_path=str(raw.get("device_path", "")),
        pixel_format=int(raw.get("pixel_format", 0)),
        frame_width=int(raw.get("frame_width", 0)),
        frame_height=int(raw.get("frame_height", 0)),
        interval_numerator=int(raw.get("interval_numerator", 0)),
        interval_denominator=int(raw.get("interval_denominator", 0)),
        control_values={
            str(key): value
            for key, value in control_values.items()
            if isinstance(value, bool | int | str)
        },
    )


def fourcc_to_str(pixel_format: int) -> str:
    code = pixel_format & 0x7FFFFFFF
    text = "".join(chr((code >> (8 * idx)) & 0xFF) for idx in range(4))
    return text.strip()


def format_label(color_format: ColorFormat) -> str:
    tags: list[str] = []
    if color_format.is_compressed:
        tags.append("compressed")
    if color_format.is_emulated:
        tags.append("emulated")

    suffix = f" [{', '.join(tags)}]" if tags else ""
    return f"{color_format.description} ({fourcc_to_str(color_format.pixelformat)}){suffix}"


def frame_size_label(frame_size: FrameSize) -> str:
    return f"{frame_size.width} x {frame_size.height}"


def frame_interval_fps(interval: FrameInterval) -> float:
    if interval.numerator <= 0:
        return 0.0
    return interval.denominator / interval.numerator


def frame_interval_label(interval: FrameInterval) -> str:
    fps = frame_interval_fps(interval)
    if fps <= 0:
        return "Unknown"
    if abs(fps - round(fps)) < 0.01:
        return f"{round(fps)} fps"
    return f"{fps:.2f} fps"


def unique_frame_intervals(intervals: list[FrameInterval]) -> list[FrameInterval]:
    seen: set[tuple[int, int]] = set()
    unique: list[FrameInterval] = []
    for interval in intervals:
        key = (int(interval.numerator), int(interval.denominator))
        if key in seen:
            continue
        seen.add(key)
        unique.append(interval)
    return unique


def humanize_control_name(name: str) -> str:
    return name.replace("_", " ").strip().title()


def video_path_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.removeprefix("video")
    index = int(suffix) if suffix.isdecimal() else 1_000_000
    return index, path.name


def list_video_devices() -> list[VideoDeviceInfo]:
    devices: list[VideoDeviceInfo] = []
    for path in sorted(Path("/dev").glob("video*"), key=video_path_sort_key):
        if not path.is_char_device():
            continue
        try:
            dev = Device(path)
        except Exception as exc:
            L.warning(f"Skipping {path}: {exc.__class__.__name__}({exc})")
            continue
        if not dev.is_video_capture_capable:
            continue
        devices.append(VideoDeviceInfo(str(path), dev.device_name, dev.driver_name))
    return devices


def supported_formats(device: Device) -> list[ColorFormat]:
    return [
        color_format
        for color_format in device.available_formats.keys()
        if color_format.pixelformat in SUPPORTED_PIXEL_FORMATS
    ]


def find_color_format(device: Device, pixel_format: int) -> ColorFormat:
    for color_format in device.available_formats.keys():
        if color_format.pixelformat == pixel_format:
            return color_format
    raise ValueError(f"Pixel format {fourcc_to_str(pixel_format)} is not available")


def choose_color_format(device: Device, settings: Settings) -> ColorFormat:
    formats = supported_formats(device)
    if not formats:
        available = ", ".join(
            f"{fmt.description} ({fourcc_to_str(fmt.pixelformat)})"
            for fmt in device.available_formats.keys()
        )
        raise ValueError(
            f"No supported pixel format found for {device.path}. Available: {available}"
        )

    if settings.pixel_format:
        for color_format in formats:
            if color_format.pixelformat == settings.pixel_format:
                return color_format

    try:
        current_format, _current_size = device.get_format()
        for color_format in formats:
            if color_format.pixelformat == current_format.pixelformat:
                return color_format
    except Exception:
        pass

    return formats[0]


def choose_frame_size(device: Device, color_format: ColorFormat, settings: Settings) -> FrameSize:
    sizes = device.available_formats[color_format]
    if not sizes:
        raise ValueError(f"No frame sizes available for {format_label(color_format)}")

    for frame_size in sizes:
        if frame_size.width == settings.frame_width and frame_size.height == settings.frame_height:
            return frame_size

    try:
        current_format, current_size = device.get_format()
        if current_format.pixelformat == color_format.pixelformat:
            for frame_size in sizes:
                if frame_size == current_size:
                    return frame_size
    except Exception:
        pass

    return sizes[0]


def choose_frame_interval(
    device: Device, color_format: ColorFormat, frame_size: FrameSize, settings: Settings
) -> FrameInterval:
    try:
        intervals = device.get_available_frame_intervals(color_format, frame_size)
    except Exception:
        intervals = []
    intervals = unique_frame_intervals(intervals)
    if not intervals:
        with contextlib.suppress(Exception):
            return device.get_frame_interval()
        return FrameInterval()

    for interval in intervals:
        if (
            interval.numerator == settings.interval_numerator
            and interval.denominator == settings.interval_denominator
        ):
            return interval

    with contextlib.suppress(Exception):
        current_interval = device.get_frame_interval()
        for interval in intervals:
            if interval == current_interval:
                return interval

    return intervals[0]


def build_capture_config(settings: Settings) -> CaptureConfig:
    device_path = settings.device_path.strip()
    if not device_path:
        devices = list_video_devices()
        if not devices:
            raise ValueError("No V4L2 video capture device found")
        device_path = devices[0].path

    device = Device(device_path)
    if not device.is_video_capture_capable:
        raise ValueError(f"{device_path} is not a video capture device")

    color_format = choose_color_format(device, settings)
    frame_size = choose_frame_size(device, color_format, settings)
    device.set_format(color_format, frame_size)
    color_format, frame_size = device.get_format()

    interval = choose_frame_interval(device, color_format, frame_size, settings)
    if interval.numerator > 0 and interval.denominator > 0:
        with contextlib.suppress(Exception):
            device.set_frame_interval(interval)
    with contextlib.suppress(Exception):
        interval = device.get_frame_interval()

    return CaptureConfig(
        device_path=device_path,
        pixel_format=color_format.pixelformat,
        frame_width=frame_size.width,
        frame_height=frame_size.height,
        interval_numerator=interval.numerator,
        interval_denominator=interval.denominator,
        control_values=dict(settings.control_values),
    )


def find_control(device: Device, control_id: int) -> Control | None:
    for control in device.controls:
        if control.id == control_id:
            return control
    return None


def menu_item_by_index(menu: Menu, index: int) -> Item | None:
    for item in menu.items:
        if item.index == index:
            return item
    return None


def control_value_to_json(control: Control, value: Any) -> JsonControlValue:
    if isinstance(control, Menu):
        if isinstance(value, Item):
            return int(value.index)
        return int(value)
    if control.type == V4L2_CTRL_TYPE_BOOLEAN:
        return bool(value)
    if control.type in (V4L2_CTRL_TYPE_INTEGER, V4L2_CTRL_TYPE_INTEGER64, V4L2_CTRL_TYPE_BITMASK):
        return int(value)
    if control.type == V4L2_CTRL_TYPE_STRING:
        return str(value)
    raise ValueError(f"Unsupported control type {control.type} for {control.name}")


def auto_control_is_enabled(control_id: int, value: JsonControlValue) -> bool:
    if control_id == V4L2_CID_EXPOSURE_AUTO:
        with contextlib.suppress(TypeError, ValueError):
            return int(value) != V4L2_EXPOSURE_MANUAL
    return bool(value)


def normalized_control_name(control: Control) -> str:
    return control.name.lower().replace(" ", "_").replace("-", "_").replace(",", "")


def control_apply_priority(control: Control) -> tuple[int, int]:
    if control.id in AUTO_CONTROL_IDS:
        return 0, control.id
    if control.id == V4L2_CID_EXPOSURE_AUTO_PRIORITY:
        return 1, control.id
    if control.id in DEPENDENT_CONTROL_IDS:
        return 20, control.id
    return CONTROL_NAME_PRIORITIES.get(normalized_control_name(control), 10), control.id


def should_consider_control(control: Control) -> bool:
    if control.is_disabled:
        return False
    if control.is_read_only or control.is_volatile:
        return False
    if control.type in (V4L2_CTRL_TYPE_BUTTON, V4L2_CTRL_TYPE_CTRL_CLASS):
        return False
    return isinstance(control, Menu) or control.type in VALUE_CONTROL_TYPES


def should_persist_control(control: Control) -> bool:
    if not should_consider_control(control):
        return False
    if control.is_inactive:
        return False
    return True


def should_apply_saved_control(control: Control) -> bool:
    if not should_consider_control(control):
        return False
    if control.is_inactive and control.id not in DEPENDENT_CONTROL_IDS:
        return False
    return True


def should_apply_live_control(control: Control) -> bool:
    if control.is_disabled or control.is_read_only or control.is_inactive:
        return False
    if control.type == V4L2_CTRL_TYPE_CTRL_CLASS:
        return False
    return (
        isinstance(control, Menu)
        or control.type in VALUE_CONTROL_TYPES
        or control.type == V4L2_CTRL_TYPE_BUTTON
    )


def read_camera_control_values(device: Device) -> dict[str, JsonControlValue]:
    values: dict[str, JsonControlValue] = {}
    for control in device.controls:
        if not should_persist_control(control):
            continue
        try:
            values[str(control.id)] = control_value_to_json(
                control,
                device.get_control_value(control),
            )
        except Exception as exc:
            L.error(
                f"Unable to read control {control.name} "
                + f"(0x{control.id:08x}): {exc.__class__.__name__}({exc})"
            )
    return values


def value_for_control_set(control: Control, value: JsonControlValue) -> bool | int | str | Item:
    if isinstance(control, Menu):
        item = menu_item_by_index(control, int(value))
        if item is None:
            raise ValueError(f"Invalid menu index {value} for {control.name}")
        return item
    if control.type in (V4L2_CTRL_TYPE_BOOLEAN, V4L2_CTRL_TYPE_BUTTON):
        return bool(value)
    if control.type in (V4L2_CTRL_TYPE_INTEGER, V4L2_CTRL_TYPE_INTEGER64, V4L2_CTRL_TYPE_BITMASK):
        return int(value)
    if control.type == V4L2_CTRL_TYPE_STRING:
        return str(value)
    raise ValueError(f"Unsupported control type {control.type} for {control.name}")


def apply_control_update(device: Device, update: ControlUpdate) -> None:
    control = find_control(device, update.control_id)
    if control is None:
        return
    if not should_apply_live_control(control):
        return
    device.set_control_value(control, value_for_control_set(control, update.value))


def apply_saved_controls(device: Device, control_values: dict[str, JsonControlValue]) -> None:
    pending: list[tuple[tuple[int, int], ControlUpdate]] = []
    for key, value in control_values.items():
        try:
            control_id = int(key)
        except ValueError:
            continue
        control = find_control(device, control_id)
        if control is None or not should_consider_control(control):
            continue
        pending.append((control_apply_priority(control), ControlUpdate(control_id, value)))

    for _priority, update in sorted(pending, key=lambda item: item[0]):
        control = find_control(device, update.control_id)
        if control is None or not should_apply_saved_control(control):
            continue
        try:
            device.set_control_value(control, value_for_control_set(control, update.value))
        except Exception as exc:
            L.error(f"Unable to set control {update.control_id}: {exc.__class__.__name__}({exc})")


def drain_control_queue(device: Device, control_queue: ControlUpdateBuffer) -> None:
    updates = control_queue.drain()
    updates.sort(
        key=lambda update: (
            control_apply_priority(control)
            if (control := find_control(device, update.control_id)) is not None
            else (10, update.control_id)
        )
    )
    for update in updates:
        try:
            apply_control_update(device, update)
        except Exception as exc:
            L.error(f"Live control update failed: {exc.__class__.__name__}({exc})")


def queue_put_drop_oldest(
    frame_queue: queue.Queue[CapturedFrame | CaptureError], item: CapturedFrame | CaptureError
) -> bool:
    try:
        frame_queue.put_nowait(item)
        return False
    except queue.Full:
        pass

    if isinstance(item, CaptureError):
        with contextlib.suppress(queue.Empty):
            _ = frame_queue.get_nowait()
        with contextlib.suppress(queue.Full):
            frame_queue.put_nowait(item)
        return True

    try:
        dropped_item = frame_queue.get_nowait()
    except queue.Empty:
        dropped_item = None

    if isinstance(dropped_item, CaptureError):
        with contextlib.suppress(queue.Full):
            frame_queue.put_nowait(dropped_item)
        return True

    try:
        frame_queue.put_nowait(item)
    except queue.Full:
        return True
    return isinstance(dropped_item, CapturedFrame)


def trim_jpeg_payload(frame_bytes: bytes) -> bytes:
    start = frame_bytes.find(b"\xff\xd8")
    if start < 0:
        return frame_bytes

    end = frame_bytes.rfind(b"\xff\xd9")
    if end >= start:
        return frame_bytes[start : end + 2]

    return frame_bytes[start:]


def decode_frame(frame_bytes: bytes, pixel_format: int, width: int, height: int) -> np.ndarray:
    if pixel_format in (V4L2_PIX_FMT_MJPEG, V4L2_PIX_FMT_JPEG):
        raw = np.frombuffer(trim_jpeg_payload(bytes(frame_bytes)), dtype=np.uint8)
        mat = cv.imdecode(raw, cv.IMREAD_COLOR)
        if mat is None:
            raise FrameDecodeError("OpenCV failed to decode JPEG frame")
        return mat

    if pixel_format == V4L2_PIX_FMT_YUYV:
        raw = np.frombuffer(frame_bytes[: width * height * 2], dtype=np.uint8)
        yuyv = raw.reshape((height, width, 2))
        return cv.cvtColor(yuyv, cv.COLOR_YUV2BGR_YUY2)

    if pixel_format == V4L2_PIX_FMT_UYVY:
        raw = np.frombuffer(frame_bytes[: width * height * 2], dtype=np.uint8)
        uyvy = raw.reshape((height, width, 2))
        return cv.cvtColor(uyvy, cv.COLOR_YUV2BGR_UYVY)

    if pixel_format == V4L2_PIX_FMT_RGB24:
        raw = np.frombuffer(frame_bytes[: width * height * 3], dtype=np.uint8)
        rgb = raw.reshape((height, width, 3))
        return cv.cvtColor(rgb, cv.COLOR_RGB2BGR)

    if pixel_format == V4L2_PIX_FMT_BGR24:
        raw = np.frombuffer(frame_bytes[: width * height * 3], dtype=np.uint8)
        return raw.reshape((height, width, 3)).copy()

    if pixel_format == V4L2_PIX_FMT_GREY:
        raw = np.frombuffer(frame_bytes[: width * height], dtype=np.uint8)
        return raw.reshape((height, width)).copy()

    if pixel_format == V4L2_PIX_FMT_Y16:
        raw = np.frombuffer(frame_bytes[: width * height * 2], dtype=np.uint16)
        return raw.reshape((height, width)).copy()

    raise ValueError(f"Unsupported pixel format: {fourcc_to_str(pixel_format)}")


def capture_loop(
    config: CaptureConfig,
    frame_queue: queue.Queue[CapturedFrame | CaptureError],
    control_queue: ControlUpdateBuffer,
    stop_event: threading.Event,
) -> None:
    try:
        device = Device(config.device_path)
        color_format = find_color_format(device, config.pixel_format)
        device.set_format(color_format, FrameSize(config.frame_width, config.frame_height))

        if config.interval_numerator > 0 and config.interval_denominator > 0:
            with contextlib.suppress(Exception):
                device.set_frame_interval(
                    FrameInterval(config.interval_numerator, config.interval_denominator)
                )

        apply_saved_controls(device, config.control_values)

        last_sequence: int | None = None
        decode_failures = 0
        v4l2_error_count = 0
        v4l2_sequence_warning_count = 0
        warned_timestamp_kinds: set[tuple[int, int]] = set()
        queue_drop_count = 0
        for stream_frame in Stream(device).iter_frames():
            drain_control_queue(device, control_queue)
            if stop_event.is_set():
                break

            metadata = stream_frame.metadata
            if last_sequence is not None:
                sequence_gap = v4l2_sequence_gap(last_sequence, metadata.sequence)
                if sequence_gap is None:
                    v4l2_sequence_warning_count += 1
                    if (
                        v4l2_sequence_warning_count == 1
                        or v4l2_sequence_warning_count % DROP_WARNING_INTERVAL == 0
                    ):
                        L.warning(
                            "V4L2 buffer sequence moved unexpectedly: "
                            + f"previous={last_sequence}, current={metadata.sequence}"
                        )
                elif sequence_gap > 0:
                    v4l2_sequence_warning_count += 1
                    if (
                        v4l2_sequence_warning_count == 1
                        or v4l2_sequence_warning_count % DROP_WARNING_INTERVAL == 0
                    ):
                        L.warning(
                            "V4L2 buffer sequence gap detected: "
                            + f"missed {sequence_gap} buffer(s); "
                            + f"previous={last_sequence}, current={metadata.sequence}"
                        )
            last_sequence = metadata.sequence

            timestamp_kind = (metadata.timestamp_type, metadata.timestamp_source)
            if (
                metadata.timestamp_type != V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC
                and timestamp_kind not in warned_timestamp_kinds
            ):
                warned_timestamp_kinds.add(timestamp_kind)
                L.warning(
                    "V4L2 buffer timestamp is not CLOCK_MONOTONIC; "
                    + "submitting the raw driver timestamp. "
                    + f"type={v4l2_timestamp_type_name(metadata.timestamp_type)}, "
                    + f"source={v4l2_timestamp_source_name(metadata.timestamp_source)}, "
                    + f"flags=0x{metadata.flags:08x}"
                )

            if metadata.has_error:
                v4l2_error_count += 1
                if v4l2_error_count == 1 or v4l2_error_count % DROP_WARNING_INTERVAL == 0:
                    L.warning(
                        "Dropping V4L2 buffer flagged with ERROR: "
                        + f"sequence={metadata.sequence}, "
                        + f"timestamp_us={metadata.timestamp_us}, "
                        + f"bytesused={metadata.bytesused}, "
                        + f"flags=0x{metadata.flags:08x}"
                    )
                continue

            frame_index = int(metadata.sequence)
            frame_time_us = v4l2_timestamp_to_syntalos_us(metadata, config)
            try:
                mat = decode_frame(
                    stream_frame.data,
                    config.pixel_format,
                    config.frame_width,
                    config.frame_height,
                )
            except FrameDecodeError as exc:
                decode_failures += 1
                if decode_failures == 1 or decode_failures % DECODE_WARNING_INTERVAL == 0:
                    L.warning(f"Skipping undecodable frame {frame_index}: {exc}")
                continue

            if queue_put_drop_oldest(frame_queue, CapturedFrame(frame_index, frame_time_us, mat)):
                queue_drop_count += 1
                if queue_drop_count == 1 or queue_drop_count % DROP_WARNING_INTERVAL == 0:
                    L.warning(
                        f"Capture queue full; dropped {queue_drop_count} frame(s) before submission"
                    )

            if stop_event.is_set():
                break
    except Exception as exc:
        if stop_event.is_set():
            return
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        _ = queue_put_drop_oldest(frame_queue, CaptureError(detail))


def clear_layout(layout: QFormLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.deleteLater()
        if child_layout is not None:
            while child_layout.count():
                child = child_layout.takeAt(0)
                child_widget = child.widget()
                if child_widget is not None:
                    child_widget.deleteLater()


def control_tooltip(control: Control) -> str:
    flags: list[str] = []
    if control.is_read_only:
        flags.append("read-only")
    if control.is_inactive:
        flags.append("inactive")
    if control.is_volatile:
        flags.append("volatile")
    flags_text = f", flags: {', '.join(flags)}" if flags else ""
    return (
        f"id: 0x{control.id:08x}, default: {control.default_value}, "
        f"range: {control.minimum}..{control.maximum}, step: {control.step}{flags_text}"
    )


def item_label(item: Item) -> str:
    if isinstance(item, MenuItem):
        return item.name
    if isinstance(item, IntegerMenuItem):
        return str(item.value)
    return str(item.index)


def control_step(control: Control) -> int:
    return max(int(control.step), 1)


def control_step_count(control: Control) -> int:
    minimum = int(control.minimum)
    maximum = int(control.maximum)
    return max((maximum - minimum) // control_step(control), 0)


def control_value_from_step_index(control: Control, index: int) -> int:
    minimum = int(control.minimum)
    return minimum + max(index, 0) * control_step(control)


def control_step_index_for_value(control: Control, value: int) -> int:
    minimum = int(control.minimum)
    maximum_steps = control_step_count(control)
    raw_index = round((max(int(value), minimum) - minimum) / control_step(control))
    return max(0, min(maximum_steps, raw_index))


def snap_control_value(control: Control, value: int) -> int:
    return control_value_from_step_index(control, control_step_index_for_value(control, value))


@final
class Module:
    def __init__(self, mlink: syl.SyntalosLink, app: QApplication) -> None:
        self.mlink = mlink
        self.app = app
        self.settings = Settings()

        self.running = False
        self.capture_config: CaptureConfig | None = None
        self.capture_thread: threading.Thread | None = None
        self.capture_stop_event: threading.Event | None = None
        self.frame_queue: queue.Queue[CapturedFrame | CaptureError] | None = None
        self.control_queue: ControlUpdateBuffer | None = None
        self.next_expected_frame_index: int | None = None
        self.dropped_frame_count = 0
        self.frame_gap_warning_count = 0

        self.settings_dialog: QDialog | None = None
        self.populating_controls = False
        self.capture_widgets_enabled: bool | None = None
        self.control_widgets: dict[int, QWidget] = {}
        self.control_widget_base_enabled: dict[int, bool] = {}
        self.control_value_setters: dict[int, ControlValueSetter] = {}
        self.control_display_values: dict[int, JsonControlValue] = {}

        self.register_ports()
        self.register_callbacks()

    # # ################################################################################
    # # Syntalos interface
    # # ################################################################################

    def register_ports(self) -> None:
        self.out_frames = self.mlink.register_output_port("frames", "Frames", syl.DataType.Frame)

    def register_callbacks(self) -> None:
        self.mlink.on_prepare = self.prepare
        self.mlink.on_start = self.start
        self.mlink.on_stop = self.stop
        self.mlink.on_show_settings = self.show_settings
        self.mlink.on_save_settings = self.save_settings
        self.mlink.on_load_settings = self.load_settings

    def prepare(self) -> bool:
        self.stop_capture_thread()
        self.running = False

        config = build_capture_config(self.settings)
        self.capture_config = config
        self.settings.device_path = config.device_path
        self.settings.pixel_format = config.pixel_format
        self.settings.frame_width = config.frame_width
        self.settings.frame_height = config.frame_height
        self.settings.interval_numerator = config.interval_numerator
        self.settings.interval_denominator = config.interval_denominator

        self.out_frames.set_metadata_value_size("size", [config.frame_width, config.frame_height])
        self.out_frames.set_metadata_value("pixel_format", fourcc_to_str(config.pixel_format))
        self.out_frames.set_metadata_value("time_source", "v4l2_buffer_timestamp")
        self.out_frames.set_metadata_value("index_source", "v4l2_buffer_sequence")
        fps = frame_interval_fps(
            FrameInterval(config.interval_numerator, config.interval_denominator)
        )
        if fps > 0:
            self.out_frames.set_metadata_value("framerate", fps)

        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAX)
        self.control_queue = ControlUpdateBuffer()
        self.capture_stop_event = threading.Event()
        self.update_capture_widget_state()
        return True

    def start(self) -> None:
        if self.capture_config is None:
            raise RuntimeError("Camera was not prepared")

        assert self.frame_queue is not None
        assert self.control_queue is not None
        assert self.capture_stop_event is not None

        start_monotonic_ns, start_syl_us = syntalos_clock_anchor()
        config = replace(
            self.capture_config,
            start_monotonic_ns=start_monotonic_ns,
            start_syl_us=start_syl_us,
        )
        self.capture_config = config
        self.running = True
        self.next_expected_frame_index = None
        self.dropped_frame_count = 0
        self.frame_gap_warning_count = 0
        self.update_capture_widget_state()

        self.capture_thread = threading.Thread(
            target=capture_loop,
            args=(config, self.frame_queue, self.control_queue, self.capture_stop_event),
            name="uvc-camera-capture",
            daemon=True,
        )
        self.capture_thread.start()

    def event_loop_tick(self) -> None:
        self.app.processEvents()
        self.process_capture_queue()
        self.update_capture_widget_state_if_needed()

        if self.running and self.capture_thread is not None and not self.capture_thread.is_alive():
            self.running = False
            self.update_capture_widget_state()
            raise RuntimeError("Camera capture thread stopped unexpectedly")

    def stop(self) -> None:
        self.running = False
        self.stop_capture_thread()
        self.update_capture_widget_state()

    def load_settings(self, settings: bytes, _base_dir: Path) -> bool:
        try:
            self.settings = deserialise_settings(settings)
            return True
        except Exception:
            self.settings = Settings()
            raise

    def save_settings(self, _base_dir: Path) -> bytes:
        self.update_settings_from_camera()
        return serialise_settings(self.settings)

    def update_settings_from_camera(self) -> None:
        device_path = self.settings.device_path.strip()
        if not device_path and self.capture_config is not None:
            device_path = self.capture_config.device_path
        if not device_path:
            return

        try:
            device = Device(device_path)
            color_format, frame_size = device.get_format()
            interval = device.get_frame_interval()
            control_values = read_camera_control_values(device)
        except Exception as exc:
            L.error(
                f"Unable to read camera settings from {device_path}: "
                + f"{exc.__class__.__name__}({exc})"
            )
            return

        self.settings.device_path = device_path
        self.settings.pixel_format = int(color_format.pixelformat)
        self.settings.frame_width = int(frame_size.width)
        self.settings.frame_height = int(frame_size.height)
        self.settings.interval_numerator = int(interval.numerator)
        self.settings.interval_denominator = int(interval.denominator)
        self.settings.control_values = control_values

    # # ################################################################################
    # # Capture handling
    # # ################################################################################

    def stop_capture_thread(self) -> None:
        stop_event = self.capture_stop_event
        if stop_event is not None:
            stop_event.set()

        thread = self.capture_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():
                L.error("Camera capture thread did not stop within timeout")

        self.capture_thread = None
        self.capture_stop_event = None

    def note_submitted_frame_index(self, frame_index: int) -> bool:
        expected_index = self.next_expected_frame_index
        current_index = frame_index % V4L2_SEQUENCE_MODULUS
        if expected_index is None:
            self.next_expected_frame_index = (current_index + 1) % V4L2_SEQUENCE_MODULUS
            return True

        delta = (current_index - expected_index) % V4L2_SEQUENCE_MODULUS
        if 0 < delta < (V4L2_SEQUENCE_MODULUS // 2):
            missed = delta
            self.dropped_frame_count += missed
            self.frame_gap_warning_count += 1
            if (
                self.frame_gap_warning_count == 1
                or self.frame_gap_warning_count % DROP_WARNING_INTERVAL == 0
            ):
                L.warning(
                    "Camera frame index gap detected: "
                    + f"missed {missed} frame(s); "
                    + f"{self.dropped_frame_count} total frame(s) missed"
                )
            self.set_status(f"Recording; {self.dropped_frame_count} frame(s) missed.")
        elif delta >= (V4L2_SEQUENCE_MODULUS // 2):
            L.warning(
                f"Ignoring out-of-order camera frame index {frame_index}, expected {expected_index}"
            )
            return False

        self.next_expected_frame_index = (current_index + 1) % V4L2_SEQUENCE_MODULUS
        return True

    def process_capture_queue(self) -> None:
        frame_queue = self.frame_queue
        if frame_queue is None:
            return

        while True:
            try:
                item = frame_queue.get_nowait()
            except queue.Empty:
                break

            if isinstance(item, CaptureError):
                self.running = False
                self.update_capture_widget_state()
                raise RuntimeError(f"Camera capture failed: {item.message}")

            if not self.running:
                continue

            if not self.note_submitted_frame_index(item.index):
                continue
            frame = syl.Frame()
            frame.mat = item.mat
            frame.time_usec = item.time_usec
            frame.index = item.index
            self.out_frames.submit(frame)

    def queue_control_update(self, control_id: int, value: JsonControlValue) -> None:
        if self.running and self.control_queue is not None:
            self.control_queue.put(ControlUpdate(control_id, value))

    # # ################################################################################
    # # Settings GUI
    # # ################################################################################

    def show_settings(self) -> None:
        dialog = self.settings_dialog
        if dialog is not None:
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            return

        dialog = self.build_settings_dialog()
        self.settings_dialog = dialog
        self.refresh_device_list()

        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def build_settings_dialog(self) -> QDialog:
        dialog = QDialog()
        dialog.setWindowTitle("UVC Camera Settings")
        dialog.resize(760, 640)

        main_layout = QVBoxLayout(dialog)

        capture_group = QGroupBox("Capture")
        capture_layout = QFormLayout(capture_group)
        capture_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        device_row = QHBoxLayout()
        dialog.deviceComboBox = QComboBox()
        dialog.deviceComboBox.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        dialog.refreshDevicesButton = QPushButton()
        dialog.refreshDevicesButton.setIcon(QIcon.fromTheme("view-refresh"))
        dialog.refreshDevicesButton.setToolTip("Refresh devices")
        device_row.addWidget(dialog.deviceComboBox, 1)
        device_row.addWidget(dialog.refreshDevicesButton)
        capture_layout.addRow("Device", device_row)

        dialog.formatComboBox = QComboBox()
        dialog.sizeComboBox = QComboBox()
        dialog.intervalComboBox = QComboBox()
        capture_layout.addRow("Format", dialog.formatComboBox)
        capture_layout.addRow("Frame size", dialog.sizeComboBox)
        capture_layout.addRow("Frame rate", dialog.intervalComboBox)
        main_layout.addWidget(capture_group)

        controls_group = QGroupBox("Camera Controls")
        controls_layout = QVBoxLayout(controls_group)

        control_toolbar = QHBoxLayout()
        dialog.refreshControlsButton = QPushButton("Refresh Controls")
        dialog.refreshControlsButton.setIcon(QIcon.fromTheme("view-refresh"))
        control_toolbar.addStretch(1)
        control_toolbar.addWidget(dialog.refreshControlsButton)
        controls_layout.addLayout(control_toolbar)

        dialog.controlScrollArea = QScrollArea()
        dialog.controlScrollArea.setWidgetResizable(True)
        dialog.controlContainer = QWidget()
        dialog.controlFormLayout = QFormLayout(dialog.controlContainer)
        dialog.controlFormLayout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        dialog.controlScrollArea.setWidget(dialog.controlContainer)
        controls_layout.addWidget(dialog.controlScrollArea, 1)
        main_layout.addWidget(controls_group, 1)

        dialog.statusLabel = QLabel()
        dialog.statusLabel.setWordWrap(True)
        main_layout.addWidget(dialog.statusLabel)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(dialog.reject)
        main_layout.addWidget(button_box)

        dialog.refreshDevicesButton.clicked.connect(self.refresh_device_list)
        dialog.refreshControlsButton.clicked.connect(self.refresh_controls)
        dialog.deviceComboBox.currentIndexChanged.connect(self.on_device_changed)
        dialog.formatComboBox.currentIndexChanged.connect(self.on_format_changed)
        dialog.sizeComboBox.currentIndexChanged.connect(self.on_size_changed)
        dialog.intervalComboBox.currentIndexChanged.connect(self.on_interval_changed)
        dialog.finished.connect(self.cleanup_settings_dialog)

        self.update_capture_widget_state(dialog)
        return dialog

    def cleanup_settings_dialog(self, _result: int) -> None:
        dialog = self.settings_dialog
        self.settings_dialog = None
        self.clear_control_widget_state()
        if dialog is not None:
            dialog.deleteLater()

    def update_capture_widget_state(self, dialog: QDialog | None = None) -> None:
        dialog = dialog or self.settings_dialog
        if dialog is None:
            return

        capture_enabled = not self.running and not self.mlink.is_running
        self.capture_widgets_enabled = capture_enabled
        for attr in (
            "deviceComboBox",
            "refreshDevicesButton",
            "formatComboBox",
            "sizeComboBox",
            "intervalComboBox",
        ):
            widget = getattr(dialog, attr, None)
            if widget is not None:
                widget.setEnabled(capture_enabled)

    def update_capture_widget_state_if_needed(self) -> None:
        dialog = self.settings_dialog
        if dialog is None:
            return

        capture_enabled = not self.running and not self.mlink.is_running
        previous_enabled = self.capture_widgets_enabled
        if capture_enabled == previous_enabled:
            return

        self.update_capture_widget_state(dialog)
        if capture_enabled and previous_enabled is False:
            self.on_device_changed()

    def set_status(self, text: str) -> None:
        dialog = self.settings_dialog
        if dialog is not None:
            dialog.statusLabel.setText(text)

    def refresh_device_list(self) -> None:
        dialog = self.settings_dialog
        if dialog is None:
            return

        current_path = self.settings.device_path or dialog.deviceComboBox.currentData()
        dialog.deviceComboBox.blockSignals(True)
        dialog.deviceComboBox.clear()

        devices = list_video_devices()
        for info in devices:
            dialog.deviceComboBox.addItem(f"{info.name} ({info.path})", info.path)

        device_paths = {info.path for info in devices}
        if self.settings.device_path and self.settings.device_path not in device_paths:
            dialog.deviceComboBox.addItem(
                f"Saved device ({self.settings.device_path})",
                self.settings.device_path,
            )

        selected_index = dialog.deviceComboBox.findData(current_path)
        if selected_index < 0 and dialog.deviceComboBox.count() > 0:
            selected_index = 0
        dialog.deviceComboBox.setCurrentIndex(selected_index)
        dialog.deviceComboBox.blockSignals(False)

        if selected_index < 0:
            self.clear_capture_combos()
            self.clear_controls()
            self.set_status("No V4L2 video capture device found.")
            return

        self.on_device_changed()

    def selected_device_path(self) -> str:
        dialog = self.settings_dialog
        if dialog is None:
            return self.settings.device_path
        data = dialog.deviceComboBox.currentData()
        return str(data) if data is not None else ""

    def clear_capture_combos(self) -> None:
        dialog = self.settings_dialog
        if dialog is None:
            return
        for combo in (dialog.formatComboBox, dialog.sizeComboBox, dialog.intervalComboBox):
            combo.blockSignals(True)
            combo.clear()
            combo.blockSignals(False)

    def clear_controls(self) -> None:
        self.clear_control_widget_state()
        dialog = self.settings_dialog
        if dialog is None:
            return
        clear_layout(dialog.controlFormLayout)

    def clear_control_widget_state(self) -> None:
        self.control_widgets.clear()
        self.control_widget_base_enabled.clear()
        self.control_value_setters.clear()
        self.control_display_values.clear()

    def register_control_widget(
        self,
        control: Control,
        widget: QWidget,
        value: JsonControlValue,
        set_value: ControlValueSetter | None = None,
    ) -> QWidget:
        self.control_widgets[control.id] = widget
        self.control_widget_base_enabled[control.id] = not control.is_read_only
        self.control_display_values[control.id] = value
        if set_value is not None:
            self.control_value_setters[control.id] = set_value
        return widget

    def update_control_widget_value(
        self, control_id: int, value: JsonControlValue, persist: bool = True
    ) -> None:
        self.control_display_values[control_id] = value
        if persist:
            self.settings.control_values[str(control_id)] = value

        setter = self.control_value_setters.get(control_id)
        if setter is None:
            return

        try:
            setter(value)
        except Exception as exc:
            L.error(
                f"Unable to update control widget {control_id}: "
                + f"{exc.__class__.__name__}({exc})"
            )

    def update_auto_dependent_control_widgets(self) -> None:
        for auto_control_id, dependent_control_id in AUTO_CONTROL_DEPENDENT_IDS.items():
            dependent_widget = self.control_widgets.get(dependent_control_id)
            if dependent_widget is None:
                continue

            auto_value = self.control_display_values.get(auto_control_id)
            if auto_value is None:
                auto_value = self.settings.control_values.get(str(auto_control_id))
            if auto_value is None:
                continue

            dependent_widget.setEnabled(
                self.control_widget_base_enabled.get(dependent_control_id, True)
                and not auto_control_is_enabled(auto_control_id, auto_value)
            )

    def open_selected_control_device(self) -> Device | None:
        device_path = self.selected_device_path()
        if not device_path:
            return None
        try:
            return Device(device_path)
        except Exception as exc:
            self.set_status(f"Unable to open {device_path}: {exc}")
            return None

    def apply_control_to_device(
        self, device: Device, control_id: int, value: JsonControlValue
    ) -> bool:
        control = find_control(device, control_id)
        if control is None or control.is_disabled or control.is_read_only:
            return False

        try:
            device.set_control_value(control, value_for_control_set(control, value))
        except Exception as exc:
            L.error(
                f"Unable to set control {control.name} "
                + f"(0x{control.id:08x}): {exc.__class__.__name__}({exc})"
            )
            return False
        return True

    def update_dependent_control_from_device(
        self, device: Device, dependent_control_id: int
    ) -> None:
        control = find_control(device, dependent_control_id)
        if control is None:
            return

        with contextlib.suppress(Exception):
            control = device.update_control(control)

        try:
            value = control_value_to_json(control, device.get_control_value(control))
        except Exception as exc:
            L.error(
                f"Unable to read control {control.name} "
                + f"(0x{control.id:08x}): {exc.__class__.__name__}({exc})"
            )
            self.set_status(f"Unable to read {humanize_control_name(control.name)}: {exc}")
            return

        self.update_control_widget_value(dependent_control_id, value)

    def handle_auto_control_changed(self, control_id: int, value: JsonControlValue) -> None:
        dependent_control_id = AUTO_CONTROL_DEPENDENT_IDS.get(control_id)
        if dependent_control_id is None:
            return

        device = self.open_selected_control_device()
        if device is not None:
            self.apply_control_to_device(device, control_id, value)

        self.update_auto_dependent_control_widgets()
        if auto_control_is_enabled(control_id, value) or device is None:
            return

        self.update_dependent_control_from_device(device, dependent_control_id)
        self.update_auto_dependent_control_widgets()

    def on_device_changed(self, _index: int | None = None) -> None:
        device_path = self.selected_device_path()
        self.settings.device_path = device_path
        if not device_path:
            return

        try:
            device = Device(device_path)
        except Exception as exc:
            self.clear_capture_combos()
            self.clear_controls()
            self.set_status(f"Unable to open {device_path}: {exc}")
            return

        if self.running or self.mlink.is_running:
            self.populate_active_capture_combos(device)
            self.populate_controls(device)
            self.set_status(f"{device.device_name} via {device.driver_name}")
            return

        self.populate_format_combo(device)
        self.populate_controls(device)
        self.set_status(f"{device.device_name} via {device.driver_name}")

    def populate_format_combo(self, device: Device) -> None:
        dialog = self.settings_dialog
        if dialog is None:
            return

        formats = supported_formats(device)
        dialog.formatComboBox.blockSignals(True)
        dialog.formatComboBox.clear()
        for color_format in formats:
            dialog.formatComboBox.addItem(format_label(color_format), color_format.pixelformat)

        selected_index = dialog.formatComboBox.findData(self.settings.pixel_format)
        if selected_index < 0:
            with contextlib.suppress(Exception):
                selected_index = dialog.formatComboBox.findData(device.get_format()[0].pixelformat)
        if selected_index < 0 and dialog.formatComboBox.count() > 0:
            selected_index = 0
        dialog.formatComboBox.setCurrentIndex(selected_index)
        dialog.formatComboBox.blockSignals(False)

        if selected_index < 0:
            self.clear_capture_combos()
            self.set_status(f"No supported frame format found for {device.path}.")
            return

        self.settings.pixel_format = int(dialog.formatComboBox.currentData())
        self.populate_size_combo(device)

    def active_capture_config_for_device(self, device_path: str) -> CaptureConfig | None:
        config = self.capture_config
        if config is not None and config.device_path == device_path:
            return config
        if self.settings.device_path == device_path and self.settings.pixel_format:
            return CaptureConfig(
                device_path=self.settings.device_path,
                pixel_format=self.settings.pixel_format,
                frame_width=self.settings.frame_width,
                frame_height=self.settings.frame_height,
                interval_numerator=self.settings.interval_numerator,
                interval_denominator=self.settings.interval_denominator,
                control_values=dict(self.settings.control_values),
            )
        return None

    def populate_active_capture_combos(self, device: Device) -> None:
        dialog = self.settings_dialog
        if dialog is None:
            return

        config = self.active_capture_config_for_device(str(device.path))
        if config is None:
            self.clear_capture_combos()
            return

        color_format = None
        for candidate in supported_formats(device):
            if candidate.pixelformat == config.pixel_format:
                color_format = candidate
                break

        dialog.formatComboBox.blockSignals(True)
        dialog.sizeComboBox.blockSignals(True)
        dialog.intervalComboBox.blockSignals(True)
        try:
            dialog.formatComboBox.clear()
            dialog.sizeComboBox.clear()
            dialog.intervalComboBox.clear()

            if color_format is None:
                dialog.formatComboBox.addItem(
                    fourcc_to_str(config.pixel_format),
                    config.pixel_format,
                )
            else:
                dialog.formatComboBox.addItem(format_label(color_format), config.pixel_format)
            dialog.formatComboBox.setCurrentIndex(0)

            dialog.sizeComboBox.addItem(
                frame_size_label(FrameSize(config.frame_width, config.frame_height)),
                (config.frame_width, config.frame_height),
            )
            dialog.sizeComboBox.setCurrentIndex(0)

            interval = FrameInterval(config.interval_numerator, config.interval_denominator)
            dialog.intervalComboBox.addItem(
                frame_interval_label(interval),
                (config.interval_numerator, config.interval_denominator),
            )
            dialog.intervalComboBox.setCurrentIndex(0)
        finally:
            dialog.formatComboBox.blockSignals(False)
            dialog.sizeComboBox.blockSignals(False)
            dialog.intervalComboBox.blockSignals(False)

    def selected_color_format(self, device: Device) -> ColorFormat | None:
        dialog = self.settings_dialog
        if dialog is None:
            return None
        pixel_format = dialog.formatComboBox.currentData()
        if pixel_format is None:
            return None
        with contextlib.suppress(Exception):
            return find_color_format(device, int(pixel_format))
        return None

    def on_format_changed(self, _index: int | None = None) -> None:
        if self.running or self.mlink.is_running:
            return

        dialog = self.settings_dialog
        if dialog is None:
            return
        pixel_format = dialog.formatComboBox.currentData()
        if pixel_format is None:
            return
        self.settings.pixel_format = int(pixel_format)

        try:
            self.populate_size_combo(Device(self.selected_device_path()))
        except Exception as exc:
            self.set_status(f"Unable to set pixel format: {exc}")

    def populate_size_combo(self, device: Device) -> None:
        dialog = self.settings_dialog
        if dialog is None:
            return

        color_format = self.selected_color_format(device)
        if color_format is None:
            return

        sizes = device.available_formats.get(color_format, [])
        dialog.sizeComboBox.blockSignals(True)
        dialog.sizeComboBox.clear()
        for frame_size in sizes:
            dialog.sizeComboBox.addItem(
                frame_size_label(frame_size),
                (frame_size.width, frame_size.height),
            )

        selected_index = -1
        for idx in range(dialog.sizeComboBox.count()):
            width, height = dialog.sizeComboBox.itemData(idx)
            if width == self.settings.frame_width and height == self.settings.frame_height:
                selected_index = idx
                break
        if selected_index < 0:
            with contextlib.suppress(Exception):
                current_format, current_size = device.get_format()
                if current_format.pixelformat == color_format.pixelformat:
                    for idx in range(dialog.sizeComboBox.count()):
                        width, height = dialog.sizeComboBox.itemData(idx)
                        if width == current_size.width and height == current_size.height:
                            selected_index = idx
                            break
        if selected_index < 0 and dialog.sizeComboBox.count() > 0:
            selected_index = 0
        dialog.sizeComboBox.setCurrentIndex(selected_index)
        dialog.sizeComboBox.blockSignals(False)

        if selected_index >= 0:
            width, height = dialog.sizeComboBox.currentData()
            self.settings.frame_width = int(width)
            self.settings.frame_height = int(height)
            self.populate_interval_combo(device)

    def on_size_changed(self, _index: int | None = None) -> None:
        if self.running or self.mlink.is_running:
            return

        dialog = self.settings_dialog
        if dialog is None:
            return
        size_data = dialog.sizeComboBox.currentData()
        if size_data is None:
            return
        width, height = size_data
        self.settings.frame_width = int(width)
        self.settings.frame_height = int(height)

        try:
            self.populate_interval_combo(Device(self.selected_device_path()))
        except Exception as exc:
            self.set_status(f"Unable to set frame size: {exc}")

    def selected_frame_size(self) -> FrameSize | None:
        dialog = self.settings_dialog
        if dialog is None:
            return None
        size_data = dialog.sizeComboBox.currentData()
        if size_data is None:
            return None
        width, height = size_data
        return FrameSize(int(width), int(height))

    def selected_frame_interval(self) -> FrameInterval | None:
        dialog = self.settings_dialog
        if dialog is None:
            return None
        data = dialog.intervalComboBox.currentData()
        if data is None:
            return None
        numerator, denominator = data
        return FrameInterval(int(numerator), int(denominator))

    def set_combo_current_data(self, combo: QComboBox, data: Any) -> bool:
        for idx in range(combo.count()):
            if combo.itemData(idx) == data:
                previous_blocked = combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(previous_blocked)
                return True
        return False

    def sync_capture_settings_from_device(
        self,
        color_format: ColorFormat,
        frame_size: FrameSize,
        interval: FrameInterval,
    ) -> None:
        self.settings.pixel_format = int(color_format.pixelformat)
        self.settings.frame_width = int(frame_size.width)
        self.settings.frame_height = int(frame_size.height)
        self.settings.interval_numerator = int(interval.numerator)
        self.settings.interval_denominator = int(interval.denominator)

        dialog = self.settings_dialog
        if dialog is None:
            return

        self.set_combo_current_data(dialog.formatComboBox, self.settings.pixel_format)
        self.set_combo_current_data(
            dialog.sizeComboBox,
            (self.settings.frame_width, self.settings.frame_height),
        )
        self.set_combo_current_data(
            dialog.intervalComboBox,
            (self.settings.interval_numerator, self.settings.interval_denominator),
        )

    def apply_selected_capture_settings(self, device: Device) -> None:
        color_format = self.selected_color_format(device)
        frame_size = self.selected_frame_size()
        if color_format is None or frame_size is None:
            return

        device.set_format(color_format, frame_size)
        interval_error: Exception | None = None
        interval = self.selected_frame_interval()
        if interval is not None and interval.numerator > 0 and interval.denominator > 0:
            try:
                device.set_frame_interval(interval)
            except Exception as exc:
                interval_error = exc

        actual_format, actual_size = device.get_format()
        actual_interval = device.get_frame_interval()
        self.sync_capture_settings_from_device(actual_format, actual_size, actual_interval)

        if interval_error is not None:
            raise interval_error

    def populate_interval_combo(self, device: Device) -> None:
        dialog = self.settings_dialog
        if dialog is None:
            return

        color_format = self.selected_color_format(device)
        if color_format is None:
            return
        frame_size = self.selected_frame_size()
        if frame_size is None:
            return

        with contextlib.suppress(Exception):
            device.set_format(color_format, frame_size)

        try:
            intervals = device.get_available_frame_intervals(color_format, frame_size)
        except Exception:
            intervals = []
        intervals = unique_frame_intervals(intervals)
        if not intervals:
            with contextlib.suppress(Exception):
                intervals = [device.get_frame_interval()]

        dialog.intervalComboBox.blockSignals(True)
        dialog.intervalComboBox.clear()
        for interval in intervals:
            dialog.intervalComboBox.addItem(
                frame_interval_label(interval),
                (interval.numerator, interval.denominator),
            )

        selected_index = -1
        for idx in range(dialog.intervalComboBox.count()):
            numerator, denominator = dialog.intervalComboBox.itemData(idx)
            if (
                numerator == self.settings.interval_numerator
                and denominator == self.settings.interval_denominator
            ):
                selected_index = idx
                break
        if selected_index < 0:
            selected_index = 0 if dialog.intervalComboBox.count() else -1
        dialog.intervalComboBox.setCurrentIndex(selected_index)
        dialog.intervalComboBox.blockSignals(False)

        if selected_index >= 0:
            numerator, denominator = dialog.intervalComboBox.currentData()
            self.settings.interval_numerator = int(numerator)
            self.settings.interval_denominator = int(denominator)
        try:
            self.apply_selected_capture_settings(device)
        except Exception as exc:
            self.set_status(f"Unable to set capture format: {exc}")

    def on_interval_changed(self, _index: int | None = None) -> None:
        if self.running or self.mlink.is_running:
            return

        dialog = self.settings_dialog
        if dialog is None:
            return
        data = dialog.intervalComboBox.currentData()
        if data is None:
            return
        numerator, denominator = data
        self.settings.interval_numerator = int(numerator)
        self.settings.interval_denominator = int(denominator)
        try:
            self.apply_selected_capture_settings(Device(self.selected_device_path()))
        except Exception as exc:
            self.set_status(f"Unable to set frame rate: {exc}")

    def refresh_controls(self) -> None:
        dialog = self.settings_dialog
        if dialog is None:
            return
        try:
            self.populate_controls(Device(self.selected_device_path()))
        except Exception as exc:
            self.set_status(f"Unable to refresh controls: {exc}")

    def populate_controls(self, device: Device) -> None:
        dialog = self.settings_dialog
        if dialog is None:
            return

        self.populating_controls = True
        try:
            self.clear_control_widget_state()
            clear_layout(dialog.controlFormLayout)

            for control in device.controls:
                if control.is_disabled:
                    continue
                widget = self.widget_for_control(device, control)
                if widget is None:
                    continue

                label_text = humanize_control_name(control.name)
                if control.is_inactive:
                    label_text += " (Inactive)"
                label = QLabel(label_text)
                tooltip = control_tooltip(control)
                label.setToolTip(tooltip)
                widget.setToolTip(tooltip)

                if control.is_read_only:
                    widget.setEnabled(False)

                dialog.controlFormLayout.addRow(label, widget)
        finally:
            self.populating_controls = False
        self.update_auto_dependent_control_widgets()

    def widget_for_control(self, device: Device, control: Control) -> QWidget | None:
        saved = self.settings.control_values.get(str(control.id), None)
        try:
            current = control_value_to_json(control, device.get_control_value(control))
        except Exception:
            current = control.default_value
        value = saved if saved is not None else current

        if isinstance(control, Menu):
            combo = QComboBox()
            for item in control.items:
                combo.addItem(item_label(item), item.index)
            selected_index = combo.findData(int(value))
            if selected_index < 0:
                selected_index = combo.findData(int(current))
            if selected_index >= 0:
                combo.setCurrentIndex(selected_index)
            combo.currentIndexChanged.connect(
                lambda _index, ctrl_id=control.id, widget=combo: self.on_control_changed(
                    ctrl_id, int(widget.currentData())
                )
            )

            def set_combo_value(new_value: JsonControlValue, widget: QComboBox = combo) -> None:
                self.set_combo_current_data(widget, int(new_value))

            return self.register_control_widget(control, combo, int(value), set_combo_value)

        if control.type == V4L2_CTRL_TYPE_BOOLEAN:
            checkbox = QCheckBox()
            checkbox.setChecked(bool(value))
            checkbox.stateChanged.connect(
                lambda state, ctrl_id=control.id: self.on_control_changed(
                    ctrl_id, state == Qt.CheckState.Checked.value
                )
            )

            def set_checkbox_value(
                new_value: JsonControlValue, widget: QCheckBox = checkbox
            ) -> None:
                previous_blocked = widget.blockSignals(True)
                widget.setChecked(bool(new_value))
                widget.blockSignals(previous_blocked)

            return self.register_control_widget(
                control,
                checkbox,
                bool(value),
                set_checkbox_value,
            )

        if control.type == V4L2_CTRL_TYPE_BUTTON:
            button = QPushButton("Trigger")
            button.clicked.connect(
                lambda _checked=False, ctrl_id=control.id: self.on_control_changed(
                    ctrl_id, True, persist=False
                )
            )
            return button

        if control.type in (
            V4L2_CTRL_TYPE_INTEGER,
            V4L2_CTRL_TYPE_INTEGER64,
            V4L2_CTRL_TYPE_BITMASK,
        ):
            if control.minimum < SPINBOX_MIN or control.maximum > SPINBOX_MAX:
                line_edit = QLineEdit(str(value))
                line_edit.editingFinished.connect(
                    lambda ctrl_id=control.id, widget=line_edit: self.on_control_changed(
                        ctrl_id, int(widget.text())
                    )
                )

                def set_integer_text_value(
                    new_value: JsonControlValue, widget: QLineEdit = line_edit
                ) -> None:
                    previous_blocked = widget.blockSignals(True)
                    widget.setText(str(int(new_value)))
                    widget.blockSignals(previous_blocked)

                return self.register_control_widget(
                    control,
                    line_edit,
                    int(value),
                    set_integer_text_value,
                )

            return self.integer_control_widget(control, int(value))

        if control.type == V4L2_CTRL_TYPE_STRING:
            line_edit = QLineEdit(str(value))
            line_edit.editingFinished.connect(
                lambda ctrl_id=control.id, widget=line_edit: self.on_control_changed(
                    ctrl_id, widget.text()
                )
            )

            def set_string_value(
                new_value: JsonControlValue, widget: QLineEdit = line_edit
            ) -> None:
                previous_blocked = widget.blockSignals(True)
                widget.setText(str(new_value))
                widget.blockSignals(previous_blocked)

            return self.register_control_widget(
                control,
                line_edit,
                str(value),
                set_string_value,
            )

        return None

    def integer_control_widget(self, control: Control, value: int) -> QWidget:
        step_count = control_step_count(control)
        step = control_step(control)
        effective_maximum = control_value_from_step_index(control, step_count)
        initial_value = snap_control_value(control, value)

        if step_count > SPINBOX_MAX:
            spinbox = QSpinBox()
            spinbox.setRange(int(control.minimum), int(control.maximum))
            spinbox.setSingleStep(step)
            spinbox.setValue(initial_value)
            spinbox.valueChanged.connect(
                lambda new_value, ctrl_id=control.id: self.on_control_changed(
                    ctrl_id,
                    snap_control_value(control, int(new_value)),
                )
            )

            def set_spinbox_value(new_value: JsonControlValue, widget: QSpinBox = spinbox) -> None:
                previous_blocked = widget.blockSignals(True)
                widget.setValue(snap_control_value(control, int(new_value)))
                widget.blockSignals(previous_blocked)

            return self.register_control_widget(
                control,
                spinbox,
                initial_value,
                set_spinbox_value,
            )

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, step_count)
        slider.setSingleStep(1)
        slider.setPageStep(max(1, step_count // 10))
        slider.setValue(control_step_index_for_value(control, initial_value))

        spinbox = QSpinBox()
        spinbox.setRange(int(control.minimum), effective_maximum)
        spinbox.setSingleStep(step)
        spinbox.setValue(initial_value)
        spinbox.setMinimumWidth(96)

        layout.addWidget(slider, 1)
        layout.addWidget(spinbox)

        syncing = False

        def set_widgets(new_value: int) -> None:
            nonlocal syncing
            syncing = True
            slider.setValue(control_step_index_for_value(control, new_value))
            spinbox.setValue(new_value)
            syncing = False

        def submit_value(new_value: int) -> None:
            value_to_submit = snap_control_value(control, new_value)
            set_widgets(value_to_submit)
            self.on_control_changed(control.id, value_to_submit)

        def slider_changed(step_index: int) -> None:
            if syncing:
                return
            submit_value(control_value_from_step_index(control, step_index))

        def spinbox_changed(new_value: int) -> None:
            if syncing:
                return
            submit_value(new_value)

        def set_integer_value(new_value: JsonControlValue) -> None:
            set_widgets(snap_control_value(control, int(new_value)))

        slider.valueChanged.connect(slider_changed)
        spinbox.valueChanged.connect(spinbox_changed)
        return self.register_control_widget(
            control,
            container,
            initial_value,
            set_integer_value,
        )

    def on_control_changed(
        self, control_id: int, value: JsonControlValue, persist: bool = True
    ) -> None:
        if self.populating_controls:
            return

        if persist:
            self.settings.control_values[str(control_id)] = value
        self.control_display_values[control_id] = value
        self.handle_auto_control_changed(control_id, value)
        self.queue_control_update(control_id, value)


def main() -> int:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    mlink = syl.init_link(rename_process=True)
    mod = Module(mlink, app)
    try:
        mlink.await_data_forever(mod.event_loop_tick)
    finally:
        mod.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
