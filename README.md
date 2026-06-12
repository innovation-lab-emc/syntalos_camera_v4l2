# EXPERIMENTAL! Syntalos UVC Camera Module

Python Syntalos module for streaming frames from UVC/V4L2 cameras with
`pyrav4l2`.

The module opens a selected `/dev/video*` device, configures a supported pixel
format, frame size, and frame interval, decodes frames to OpenCV-compatible NumPy
arrays, and emits them on a `Frame` output port.

The settings dialog is generated in Python with PyQt6. It enumerates the selected
camera's supported formats, sizes, frame rates, and V4L2 controls. Camera controls
can remain open during a stream; live changes are sent to the capture thread via a
small queue and are intended for tuning camera settings before real acquisition.

Currently decoded formats:

- MJPEG/JPEG
- YUYV
- UYVY
- RGB24
- BGR24
- GREY
- Y16
