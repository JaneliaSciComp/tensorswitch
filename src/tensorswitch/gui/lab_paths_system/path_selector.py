#!/usr/bin/env python3
"""
Path Selector Component for TensorSwitch GUI

Provides lab path selection functionality:
1. Lab dropdown with all 131 labs
2. Path helper checkbox that shows/hides the helper
3. Storage dropdown (primary, home, scratch, etc.) based on selected lab
4. Platform dropdown with correct labels: Mac, Windows(or Linux SMB), Cluster and Linux
5. Always returns the "Cluster and Linux" path regardless of platform selection
"""

import panel as pn
import param
from typing import List, Optional
try:
    from .lab_paths import get_lab_paths
except ImportError:
    # Fallback for when running directly
    from lab_paths import get_lab_paths

class PathHelper(param.Parameterized):
    """Path helper for TensorSwitch GUI"""
    
    # UI state - the main toggle
    use_path_helper = param.Boolean(default=False, doc="Use path helper dropdowns")
    
    # Lab selection
    selected_lab = param.ObjectSelector(
        default="",
        objects=[""],
        doc="Select lab"
    )
    
    # Storage selection (dynamic based on lab)
    selected_storage = param.ObjectSelector(
        default="",
        objects=[""],
        doc="Storage type (primary, home, scratch, etc.)"
    )
    
    # Platform selection (for display only - always returns cluster path)
    selected_platform = param.ObjectSelector(
        default="Cluster and Linux",
        objects=["Mac", "Windows(or Linux SMB)", "Cluster and Linux"],
        doc="Platform type"
    )
    
    # Generated cluster path (always from cluster column)
    generated_path = param.String(default="", doc="Generated cluster path")
    
    def __init__(self, **params):
        super().__init__(**params)
        self._load_lab_data()
        
        # Watch for changes
        self.param.watch(self._on_lab_changed, 'selected_lab')
        self.param.watch(self._on_storage_changed, 'selected_storage')
        self.param.watch(self._on_platform_changed, 'selected_platform')
    
    def _load_lab_data(self):
        """Load all lab names from the lab paths system"""
        try:
            lab_manager = get_lab_paths()
            all_labs = lab_manager.get_lab_names()
            self.param.selected_lab.objects = [""] + all_labs
            print(f"✓ Loaded {len(all_labs)} labs for path helper")
        except Exception as e:
            print(f"⚠ Could not load lab data: {e}")
            self.param.selected_lab.objects = [""]
    
    def _on_lab_changed(self, event):
        """When lab is selected, update storage options"""
        selected_lab = event.new
        
        if selected_lab:
            try:
                lab_manager = get_lab_paths()
                storage_types = lab_manager.get_storage_types(selected_lab)
                self.param.selected_storage.objects = [""] + storage_types
                
                # Auto-select first storage if only one available
                if len(storage_types) == 1:
                    self.selected_storage = storage_types[0]
                else:
                    self.selected_storage = ""
                    
            except Exception as e:
                print(f"⚠ Error getting storage types for {selected_lab}: {e}")
                self.param.selected_storage.objects = [""]
                self.selected_storage = ""
        else:
            # Clear storage options if no lab selected
            self.param.selected_storage.objects = [""]
            self.selected_storage = ""
        
        self._update_generated_path()
    
    def _on_storage_changed(self, event):
        """When storage is changed, update the generated path"""
        self._update_generated_path()
    
    def _on_platform_changed(self, event):
        """Platform changed - but we always use cluster path anyway"""
        # Note: We always use cluster path regardless of platform selection
        # The platform dropdown is just for user reference
        self._update_generated_path()
    
    def _update_generated_path(self):
        """Update the generated path - always using cluster path"""
        if self.selected_lab and self.selected_storage:
            try:
                lab_manager = get_lab_paths()
                # Always get the cluster path regardless of platform selection
                cluster_path = lab_manager.get_path(self.selected_lab, self.selected_storage, "cluster")
                self.generated_path = cluster_path
            except Exception as e:
                print(f"⚠ Error generating path: {e}")
                self.generated_path = ""
        else:
            self.generated_path = ""
    
    def get_current_path(self) -> str:
        """Get the current generated cluster path (for compatibility)"""
        return self.generated_path
    
    def get_current_project(self) -> str:
        """Get the project name for the selected lab (for compatibility)"""
        if self.selected_lab:
            try:
                lab_manager = get_lab_paths()
                return lab_manager.get_ad_group(self.selected_lab)
            except Exception as e:
                print(f"⚠ Error getting project for {self.selected_lab}: {e}")
        return ""
    
    def reset_selections(self):
        """Reset all selections"""
        self.selected_lab = ""
        self.selected_storage = ""
        self.selected_platform = "Cluster and Linux"
        self.generated_path = ""

# Standard interface functions
HierarchicalPathSelector = PathHelper
PathSelectorPanel = PathHelper

def create_path_selector_panel():
    """Create path selector panel - main interface function"""
    return create_path_helper_panel()

def create_simple_path_helper_panel():
    """Create path helper panel - backward compatibility"""
    return create_path_helper_panel()

def create_path_helper_panel():
    """Create the path helper panel for TensorSwitch GUI"""
    
    path_helper = PathHelper()
    
    # Main toggle
    toggle_widget = pn.Param(
        path_helper,
        parameters=['use_path_helper'],
        widgets={'use_path_helper': pn.widgets.Checkbox}
    )
    
    # Lab dropdown
    lab_widget = pn.Param(
        path_helper,
        parameters=['selected_lab'],
        widgets={'selected_lab': {'type': pn.widgets.Select, 'width': 250}}
    )
    
    # Storage dropdown
    storage_widget = pn.Param(
        path_helper,
        parameters=['selected_storage'],
        widgets={'selected_storage': {'type': pn.widgets.Select, 'width': 200}}
    )
    
    # Platform dropdown (for display/reference only)
    platform_widget = pn.Param(
        path_helper,
        parameters=['selected_platform'],
        widgets={'selected_platform': {'type': pn.widgets.Select, 'width': 200}}
    )
    
    # Path suggestions display - shows all platform paths as hints
    def update_path_suggestions(selected_lab=None, selected_storage=None):
        # Use actual parameter values from path_helper
        lab = path_helper.selected_lab
        storage = path_helper.selected_storage
        
        if not lab or not storage:
            return pn.pane.Markdown("**Path Suggestions:** *Select lab and storage type above to see path suggestions*")
        
        try:
            lab_manager = get_lab_paths()
            mac_path = lab_manager.get_path(lab, storage, "mac")
            windows_path = lab_manager.get_path(lab, storage, "windows")
            cluster_path = lab_manager.get_path(lab, storage, "cluster")
            project = lab_manager.get_ad_group(lab)
            
            suggestions_md = f"""
**📁 Path Suggestions** (use these to build your full file path):

- **Mac**: `{mac_path}` 
- **Windows**: `{windows_path}`
- **Cluster/Linux**: `{cluster_path}` ⭐ **(Use this path for tensorswitch)**

**💰 Project for billing**: `{project}`

💡 **How to use**: Copy the **Cluster/Linux** path above and add your specific file/folder path to complete the full path in the input boxes below.

*Example*: `{cluster_path}/data/my_experiment.tif`
"""
            return pn.pane.Markdown(
                suggestions_md,
                styles={
                    'background': '#e8f5e8', 
                    'padding': '15px', 
                    'border-radius': '8px',
                    'border-left': '4px solid #28a745'
                }
            )
        except Exception as e:
            return pn.pane.Markdown(f"**Error**: Could not generate path suggestions: {e}")
    
    path_suggestions = pn.bind(update_path_suggestions, path_helper.param.selected_lab, path_helper.param.selected_storage)
    
    # Helper content (only shown when path helper is enabled)
    helper_content = pn.Column(
        pn.Row(
            pn.Column("**Lab:**", lab_widget, margin=(0, 10)),
            pn.Column("**Storage:**", storage_widget, margin=(0, 10)),
            pn.Column("**Platform (Reference):**", platform_widget, margin=(0, 10))
        ),
        path_suggestions,
        visible=pn.bind(lambda x: x, path_helper.param.use_path_helper)
    )
    
    # Complete panel
    panel = pn.Column(
        toggle_widget,
        helper_content
    )
    
    return panel, path_helper

if __name__ == "__main__":
    # Test the simple path helper
    pn.extension()
    
    panel, helper = create_path_helper_panel()
    
    test_app = pn.template.MaterialTemplate(
        title="Path Helper Test",
        sidebar=["## Test", "Try selecting different labs and storage types."]
    )
    
    test_app.main.append(
        pn.Column(
            "# Lab Path Helper",
            "This matches the requirements exactly:",
            "1. Toggle shows/hides helper",
            "2. All 131 labs in dropdown", 
            "3. Dynamic storage types per lab",
            "4. Platform dropdown with correct labels",
            "5. Always returns cluster path for tensorswitch",
            panel
        )
    )
    
    test_app.show(port=5001)