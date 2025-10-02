#!/usr/bin/env python3
"""
AI Interface for TensorSwitch
Requires OPENAI_API_KEY environment variable
"""

import panel as pn
from .ai_config import ai_config
from .tensorswitch_assistant import (
    get_tensorswitch_help_with_openai,
    get_ai_benefits,
    OPENAI_AVAILABLE
)


def create_floating_ai_chat(get_gui_state_callback=None, set_gui_params_callback=None):
    """Create floating draggable AI chat widget with interactive parameter setting (Phase 2)

    Args:
        get_gui_state_callback: Optional function that returns current GUI state as dict
        set_gui_params_callback: Optional function that sets GUI parameters from dict

    Requires OPENAI_API_KEY environment variable to be set before launching GUI
    """


    # Check if environment variable is available
    ai_available = ai_config.check_and_enable()

    # Chat widgets - larger input area
    chat_input = pn.widgets.TextAreaInput(
        placeholder="Ask about conversions, parameters, lab paths...",
        sizing_mode="stretch_width",
        height=100,
        auto_grow=True,
        max_length=1000
    )


    if ai_available:
        initial_content = """## 🤖 AI Assistant

👋 **Welcome!** I can help you convert your imaging data efficiently.

**Ask me about:**

• "I have ND2 files - how do I convert them to Zarr3?"

• "What cores should I use for a 5GB file?"

• "How do I find my lab's storage path?"

• "What's the difference between Smart and Manual mode?"

**💡 Tip:** Use Smart Mode for auto-detection!"""
    else:
        initial_content = """## 🤖 AI Assistant

❌ **Environment Variable Required**

To enable AI assistance, set your API key in terminal before launching:

```bash
export OPENAI_API_KEY="your-key-here"
pixi run python src/tensorswitch/gui/launch_gui.py
```

**What you'll get:**

• Smart conversion guidance
• Resource optimization tips
• Lab path assistance
• Parameter recommendations"""

    chat_display = pn.pane.Markdown(
        initial_content,
        sizing_mode="stretch_width",
        height=500,
        styles={
            'overflow-y': 'auto',
            'padding': '15px',
            'border': '1px solid #ddd',
            'border-radius': '5px',
            'font-size': '14px',
            'resize': 'vertical',
            'direction': 'ltr'
        }
    )

    chat_history = []
    is_minimized = False
    ai_enabled = ai_available
    pending_suggestions = {}  # Store AI parameter suggestions

    def handle_chat():
        if not ai_enabled:
            return

        user_message = chat_input.value.strip()

        if not user_message:
            return

        chat_input.value = ""
        chat_history.append(f"**👤 You:** {user_message}")
        update_display(thinking=True)

        try:
            # Phase 1: Get current GUI state for context-aware responses
            context_data = None
            if get_gui_state_callback:
                try:
                    context_data = get_gui_state_callback()
                except Exception as e:
                    print(f"Warning: Could not get GUI state: {e}")

            # Phase 2: Check if user wants to apply settings
            wants_to_apply = any(keyword in user_message.lower() for keyword in
                               ['yes', 'apply', 'apply it', 'apply them', 'do it', 'yes please', 'go ahead', 'sure'])

            # If user confirms and we have pending suggestions
            if wants_to_apply and pending_suggestions and set_gui_params_callback:
                try:
                    set_gui_params_callback(pending_suggestions)
                    param_list = ", ".join([f"{k}={v}" for k, v in pending_suggestions.items()])
                    chat_history.append(f"**✅ Applied Settings:** {param_list}")
                    chat_history.append(f"**🤖 AI:** Settings have been updated in the GUI! Please review them and click 'Run Job' when ready.")
                    pending_suggestions.clear()
                    apply_btn.visible = False
                    update_display()
                    return
                except Exception as e:
                    chat_history.append(f"**❌ Error:** {str(e)}")
                    update_display()
                    return

            # Check if user wants parameter suggestions or to change specific values
            wants_settings = any(keyword in user_message.lower() for keyword in
                               ['set parameters', 'apply settings', 'configure', 'set optimal', 'fill fields',
                                'optimal parameters', 'recommended parameters', 'suggest parameters',
                                'use', 'set', 'change to', 'make it'])

            ai_response = get_tensorswitch_help_with_openai(user_message, context_data=context_data)
            chat_history.append(f"**🤖 AI:** {ai_response}")

            # Phase 2: Extract parameters from user request first, then AI response
            if wants_settings and context_data and set_gui_params_callback:
                import re
                suggested_params = {}

                # Priority 1: Check user's message for explicit requests
                user_cores = re.search(r'(\d+)\s+cores?|cores?\s*(\d+)', user_message.lower())
                user_memory = re.search(r'(\d+)\s*gb', user_message.lower())
                user_format = re.search(r'zarr2|zarr3|n5', user_message.lower())

                if user_cores:
                    val = user_cores.group(1) or user_cores.group(2)
                    suggested_params['cores'] = str(val)
                if user_memory:
                    suggested_params['memory'] = int(user_memory.group(1))
                if user_format:
                    fmt = user_format.group(0)
                    suggested_params['output_format'] = 'Zarr3' if fmt == 'zarr3' else 'Zarr2' if fmt == 'zarr2' else 'N5'

                # Priority 2: If no explicit user request, parse AI response
                if not suggested_params:
                    cores_match = re.search(r'cores?[:\s=]+(\d+)', ai_response.lower())
                    memory_match = re.search(r'memory[:\s=]+(\d+)\s*gb', ai_response.lower())
                    time_match = re.search(r'(?:wall.?time|time)[:\s=]+(\d+):(\d+)|(\d+)\s+hours?', ai_response.lower())
                    volumes_match = re.search(r'volumes?[:\s=]+(\d+)', ai_response.lower())

                    if cores_match:
                        suggested_params['cores'] = str(cores_match.group(1))
                    if memory_match:
                        suggested_params['memory'] = int(memory_match.group(1))
                    if time_match:
                        if time_match.group(1) and time_match.group(2):
                            suggested_params['wall_time'] = f"{time_match.group(1)}:{time_match.group(2)}"
                        elif time_match.group(3):
                            suggested_params['wall_time'] = f"{time_match.group(3)}:00"
                    if volumes_match:
                        suggested_params['num_volumes'] = int(volumes_match.group(1))

                if suggested_params:
                    pending_suggestions.clear()
                    pending_suggestions.update(suggested_params)
                    param_list = ", ".join([f"{k}={v}" for k, v in suggested_params.items()])
                    chat_history.append(f"**💡 Recommended:** {param_list}")
                    chat_history.append(f"**🤖 AI:** Would you like me to apply these settings to the GUI? (Just say 'yes' or 'apply')")
                    apply_btn.visible = True

            update_display()
        except Exception as e:
            chat_history.append(f"**❌ Error:** {str(e)}")
            update_display()

    def update_display(thinking=False):
        if not ai_enabled:
            return

        content = []
        for msg in chat_history[-6:]:
            content.append(msg)
            content.append("")

        if thinking:
            content.append("**🤖 AI:** *Thinking...*")

        if not content:
            content = ["Start a conversation by asking a question!"]

        chat_display.object = "\n".join(content)

    def toggle_minimize(event):
        nonlocal is_minimized
        is_minimized = not is_minimized
        if is_minimized:
            chat_container.height = 50
            minimize_btn.name = "➕"
            chat_content.visible = False
        else:
            chat_container.height = 700  # Restore to full height
            minimize_btn.name = "➖"
            chat_content.visible = True
            # Ensure chat section is visible if AI is enabled
            if ai_enabled:
                chat_section.visible = True

    chat_btn = pn.widgets.Button(
        name="Ask AI",
        button_type="primary",
        width=80,
        height=42,
        styles={
            'font-weight': '600',
            'border-radius': '10px',
            'background': '#87CEEB !important',
            'border': 'none !important',
            'color': 'white !important',
            'box-shadow': '0 3px 10px rgba(135, 206, 235, 0.4)'
        }
    )
    # Apply gradient styling
    chat_btn.stylesheets = ["""
    .bk-btn-primary {
        background: linear-gradient(135deg, #87CEEB 0%, #4682B4 100%) !important;
        border: none !important;
    }
    """]
    chat_btn.on_click(lambda event: handle_chat())

    # Apply Settings button (Phase 2)
    apply_btn = pn.widgets.Button(
        name="Apply AI Settings",
        button_type="success",
        visible=False,
        width=150,
        height=35
    )

    def apply_settings(event):
        """Apply AI-suggested parameters to GUI"""
        if pending_suggestions and set_gui_params_callback:
            try:
                set_gui_params_callback(pending_suggestions)
                chat_history.append("**✅ Settings Applied:** Parameters updated in GUI. Please review before running.")
                apply_btn.visible = False
                pending_suggestions.clear()
                update_display()
            except Exception as e:
                chat_history.append(f"**❌ Error applying settings:** {str(e)}")
                update_display()

    apply_btn.on_click(apply_settings)

    minimize_btn = pn.widgets.Button(name="➖", button_type="light", width=30, height=25)
    minimize_btn.on_click(toggle_minimize)

    # Header with drag handle and minimize
    header = pn.Row(
        pn.pane.Markdown("🤖 **AI Assistant**", styles={'font-weight': 'bold', 'margin': '0'}),
        pn.Spacer(),
        minimize_btn,
        styles={
            'background': '#f0f0f0',
            'padding': '5px 10px',
            'border-radius': '5px 5px 0 0',
            'cursor': 'move'
        },
        sizing_mode="stretch_width",
        height=35
    )


    # Create chat section with apply button
    chat_section = pn.Column(
        pn.pane.Markdown("**Chat:**", styles={'font-size': '12px', 'margin': '2px 0'}),
        pn.Row(chat_input, chat_btn, sizing_mode="stretch_width"),
        apply_btn,  # Phase 2: Apply AI settings button
        sizing_mode="stretch_width"
    )

    # Chat content
    chat_content = pn.Column(
        chat_display,
        chat_section,
        sizing_mode="stretch_width"
    )

    # Main container with floating styles - larger size with bottom-left resize
    chat_container = pn.Column(
        header,
        chat_content,
        width=400,
        height=700,
        styles={
            'position': 'absolute',
            'top': '20px',
            'left': '20px',
            'z-index': '1000',
            'background': 'white',
            'border': '2px solid #87CEEB',
            'border-radius': '8px',
            'box-shadow': '0 4px 12px rgba(0,0,0,0.15)',
            'resize': 'both',
            'overflow': 'auto'
        }
    )

    return chat_container

def create_ai_unavailable_interface():
    """Create interface when AI is not available"""
    benefits_text = "\n".join(get_ai_benefits())

    return pn.Column(
        "## 🤖 AI Assistant",
        pn.pane.Markdown("**AI Assistant Not Available**"),
        pn.Spacer(height=10),
        pn.pane.Markdown("**To enable AI assistance:**"),
        pn.pane.Markdown("""```bash
export OPENAI_API_KEY="your-api-key-here"
pixi run python src/tensorswitch/gui/launch_gui.py
```"""),
        pn.Spacer(height=10),
        pn.pane.Markdown("**What you'll get:**"),
        pn.pane.Markdown(benefits_text),
        pn.Spacer(height=10),
        pn.pane.Markdown("**⚠️ Cost Management:**\n- Set spending limits at <a href='https://platform.openai.com' target='_blank'>platform.openai.com</a>\n- Usage: ~$0.01-0.05 per conversation"),
        sizing_mode="stretch_width"
    )

def create_openai_missing_interface():
    """Create interface when OpenAI library is missing"""
    return pn.Column(
        "## 🤖 AI Assistant",
        pn.pane.Markdown("**OpenAI library required**"),
        pn.pane.Markdown("Install with: `pip install openai`"),
        pn.pane.Markdown("Then restart TensorSwitch GUI"),
        sizing_mode="stretch_width"
    )

def create_ai_setup_widget():
    """Create AI widget - returns floating chat if environment variable is set"""
    if not OPENAI_AVAILABLE:
        return None

    # Check environment variable and return floating chat if available
    if ai_config.check_and_enable():
        return create_floating_ai_chat()

    return None


def create_floating_ai_assistant(get_gui_state_callback=None, set_gui_params_callback=None):
    """Create floating AI assistant widget with context awareness and parameter setting"""
    if not OPENAI_AVAILABLE:
        return None

    ai_config.check_and_enable()  # Check environment variable
    return create_floating_ai_chat(
        get_gui_state_callback=get_gui_state_callback,
        set_gui_params_callback=set_gui_params_callback
    )