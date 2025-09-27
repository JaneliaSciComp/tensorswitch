#!/usr/bin/env python3
"""
AI Configuration for TensorSwitch
Requires OPENAI_API_KEY environment variable
"""

import os

class AIConfig:
    """AI configuration using environment variable only"""

    def __init__(self):
        self.api_key = None
        self.is_enabled = False

    def check_and_enable(self):
        """Check environment variable and enable if key exists"""
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            self.api_key = env_key
            self.is_enabled = True
            return True
        return False


    def get_api_key(self):
        """Get the active API key"""
        return self.api_key

    def is_ai_available(self):
        """Check if AI assistant is available and enabled"""
        return self.is_enabled and self.api_key is not None



    def get_status(self):
        """Get current AI status for display"""
        if self.is_enabled:
            return "AI Assistant Ready"
        else:
            return "AI Assistant Not Available"

    def cleanup_session(self):
        """Clean up session data when GUI closes"""
        # Environment keys don't need cleanup
        pass


# Global AI configuration instance
ai_config = AIConfig()