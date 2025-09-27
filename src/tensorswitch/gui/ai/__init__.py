"""AI assistant module for TensorSwitch GUI"""

from .ai_config import ai_config
from .tensorswitch_assistant import (
    get_tensorswitch_help_with_openai,
    analyze_file_and_suggest,
    get_ai_benefits
)
from .ai_interface import create_floating_ai_assistant, create_ai_setup_widget

__all__ = [
    "ai_config",
    "get_tensorswitch_help_with_openai",
    "analyze_file_and_suggest",
    "get_ai_benefits",
    "create_floating_ai_assistant",
    "create_ai_setup_widget"
]