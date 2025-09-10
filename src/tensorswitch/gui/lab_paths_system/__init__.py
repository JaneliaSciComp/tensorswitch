"""
Lab Paths System for TensorSwitch GUI

This package provides hierarchical lab path management for HHMI labs,
including Excel parsing, hierarchical dropdowns, and project auto-population.

Retrieved: 2025-09-09
Total labs: 131
Total projects: 126
"""

from .lab_paths import (
    HierarchicalLabPaths,
    get_lab_paths,
    get_all_labs,
    get_lab_storage_types,
    get_available_platforms,
    get_lab_path,
    get_lab_project,
    get_all_projects,
    suggest_subdirs
)

from .path_selector import (
    HierarchicalPathSelector,
    PathSelectorPanel,
    create_path_selector_panel
)

__all__ = [
    'HierarchicalLabPaths',
    'get_lab_paths',
    'get_all_labs',
    'get_lab_storage_types', 
    'get_available_platforms',
    'get_lab_path',
    'get_lab_project',
    'get_all_projects',
    'suggest_subdirs',
    'HierarchicalPathSelector',
    'PathSelectorPanel',
    'create_path_selector_panel'
]