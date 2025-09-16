"""
TensorSwitch GUI Package

A Panel-based web interface for TensorSwitch data conversion tasks.
Provides an intuitive GUI for non-programmers to perform scientific data conversions.
"""

from .app import SimpleTensorSwitchGUI as TensorSwitchGUI, create_simple_app

__all__ = ["TensorSwitchGUI", "create_simple_app"]