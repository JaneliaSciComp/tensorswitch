#!/usr/bin/env python3
"""
TensorSwitch GUI - Web interface for scientific data format conversion
"""

import panel as pn
import param
import subprocess
import sys
import os
import threading
import time
import re
from pathlib import Path
import asyncio

# Import lab paths system
try:
    from lab_paths_system import get_all_projects
    from lab_paths_system.path_selector import create_simple_path_helper_panel
    LAB_PATHS_AVAILABLE = True
    print("Lab paths system loaded successfully")
except ImportError as e:
    print(f"Lab paths system not available: {e}")
    LAB_PATHS_AVAILABLE = False

# Import format detection system
FORMAT_DETECTION_AVAILABLE = False
try:
    from format_detection import FormatDetector, TaskPlanner
    FORMAT_DETECTION_AVAILABLE = True
    print("GUI: Format detection system loaded successfully")
except ImportError as e:
    print(f"GUI: Format detection system not available: {e}")
except Exception as e:
    print(f"GUI: Format detection system failed: {e}")
    import traceback
    traceback.print_exc()

# Import AI assistant system
AI_ASSISTANT_AVAILABLE = False
try:
    from ai import create_floating_ai_assistant, ai_config
    AI_ASSISTANT_AVAILABLE = True
    print("GUI: AI Assistant system loaded successfully")
except ImportError as e:
    print(f"GUI: AI Assistant system not available: {e}")
except Exception as e:
    print(f"GUI: AI Assistant system failed: {e}")
    import traceback
    traceback.print_exc()

# Import cost estimator
COST_ESTIMATOR_AVAILABLE = False
try:
    from cost_estimator import (
        estimate_processing_time,
        estimate_cluster_cost,
        get_ai_cost,
        calculate_total_chunks,
        format_time
    )
    COST_ESTIMATOR_AVAILABLE = True
    print("GUI: Cost estimator loaded successfully")
except ImportError as e:
    print(f"GUI: Cost estimator not available: {e}")
except Exception as e:
    print(f"GUI: Cost estimator failed: {e}")
    import traceback
    traceback.print_exc()

class SimpleTensorSwitchGUI(param.Parameterized):
    """TensorSwitch GUI for data format conversion"""
    
    input_path = param.String(default="", doc="Input file path")
    output_path = param.String(default="", doc="Output file path")
    task = param.ObjectSelector(
        default="tiff_to_zarr3_s0",
        objects=[
            "tiff_to_zarr3_s0",
            "nd2_to_zarr3_s0",
            "ims_to_zarr3_s0",
            "tiff_to_zarr2_s0",
            "nd2_to_zarr2_s0",
            "ims_to_zarr2_s0",
            "n5_to_zarr2",
            "n5_to_n5",
            "downsample_shard_zarr3",
            "downsample_zarr2"
        ],
        doc="Conversion task"
    )
    project = param.ObjectSelector(
        default="",
        objects=[""],  # Will be populated from lab paths data
        doc="Project for billing"
    )
    
    # Additional parameters for advanced tasks
    level = param.Integer(default=0, bounds=(0, 10), doc="Level for downsampling (0 = full resolution)")
    use_shard = param.Boolean(default=False, doc="Use sharded format for output")
    use_ome_structure = param.Boolean(default=True, doc="Use OME-Zarr multiscale structure with automatic metadata updates")
    cores = param.String(default="2", doc="Number of CPU cores for cluster jobs")
    wall_time = param.String(default="2:00", doc="Wall time for cluster jobs (HH:MM)")
    memory_limit = param.Integer(default=50, bounds=(10, 90), doc="Memory limit percentage")
    num_volumes = param.Integer(default=8, bounds=(1, 100), doc="Number of parallel volumes for cluster processing")
    run_locally = param.Boolean(default=True, doc="Run locally (unchecked = submit to cluster)")
    use_dask_jobqueue = param.Boolean(default=False, doc="Use Dask JobQueue for cluster submission (advanced)")

    # Custom shape parameters
    custom_shard_shape = param.String(default="", doc="Custom shard shape (e.g., '128,576,576') - leave empty for defaults")
    custom_chunk_shape = param.String(default="", doc="Custom chunk shape (e.g., '32,32,32') - leave empty for defaults")

    # Workflow mode selection
    workflow_mode = param.ObjectSelector(
        default="smart",
        objects=["smart", "manual"],
        doc="Workflow mode: smart (auto-detect) or manual (task selection)"
    )

    # Smart format selection parameters
    output_format = param.ObjectSelector(
        default="zarr3",
        objects=["zarr3", "zarr2", "n5"],
        doc="Output format"
    )
    max_downsample_level = param.Integer(default=0, bounds=(0, 5), doc="Maximum downsample level (0=no downsampling)")
    
    def __init__(self):
        super().__init__()
        self.current_process = None
        self.progress_thread = None
        self.job_running = False
        self.progress_data = {'total_chunks': 0, 'processed_chunks': 0, 'stage': 'Ready'}
        self.current_page = "welcome"  # Track current page: welcome, conversion, progress

        # Setup format detection
        self._setup_format_detection()

        # Setup conversion plan
        self.conversion_plan = None

        # Setup lab paths and projects
        self._setup_lab_paths_integration()

        self.create_layout()

    def _setup_format_detection(self):
        """Setup format detection system"""
        print(f"DEBUG: FORMAT_DETECTION_AVAILABLE = {FORMAT_DETECTION_AVAILABLE}")
        if FORMAT_DETECTION_AVAILABLE:
            self.format_detector = FormatDetector()
            self.task_planner = TaskPlanner()
            self.input_analysis = None
            print("Format detection system initialized")
        else:
            self.format_detector = None
            self.task_planner = None
            self.input_analysis = None
            print("Format detection system NOT available - smart mode disabled")

    def _setup_lab_paths_integration(self):
        """Setup lab paths integration and populate project dropdown"""
        if LAB_PATHS_AVAILABLE:
            try:
                # Get all project names from lab paths data
                all_projects = get_all_projects()
                self.param.project.objects = all_projects
                print(f"Populated project dropdown with {len(all_projects)} projects")
            except Exception as e:
                print(f"Could not load project data: {e}")
                # Fallback to original project list
                self.param.project.objects = ["", "ahrens", "branson", "murphy", "mengwang", "keller", "tavakoli", "scicompsoft"]
        else:
            # Fallback to original project list
            self.param.project.objects = ["", "ahrens", "branson", "murphy", "mengwang", "keller", "tavakoli", "scicompsoft"]
    
    def create_layout(self):
        """Create multi-page layout with welcome screen and conversion workflow"""
        # Create main container that will hold different pages
        self.main_container = pn.Column(sizing_mode="stretch_width")

        # Create all page layouts
        self.create_welcome_page()
        self.create_conversion_page()
        self.create_progress_page()

        # Set the layout to the main container
        self.layout = self.main_container

    def create_welcome_page(self):
        """Create the welcome/landing page"""
        # Welcome title with switch emoji
        welcome_title = pn.pane.Markdown("""
        <div style="text-align: center; margin: 60px 0;">
        <h1 style="font-size: 3.5em; margin-bottom: 20px; color: #2c3e50;">
        Welcome to TensorSwitch! 🔄
        </h1>
        <p style="font-size: 1.3em; color: #7f8c8d; margin-bottom: 40px;">
        <strong>Convert scientific data formats effortlessly</strong>
        </p>
        </div>
        """)
        
        # Explanation section
        explanation = pn.pane.Markdown("""
        <div style="background: #f8f9fa; padding: 30px; border-radius: 15px; margin: 20px 0; text-align: center;">
        <h2 style="color: #34495e; margin-bottom: 15px;">What is TensorSwitch?</h2>
        <p style="font-size: 1.1em; line-height: 1.6; color: #2c3e50; margin-bottom: 15px;">
        Convert scientific data between <strong>TIFF, ND2, IMS, N5, Zarr2, and Zarr3</strong> formats.
        No programming required.
        </p>
        <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin-top: 15px;">
        <h3 style="color: #27ae60; margin-bottom: 10px;">Key Features</h3>
        <ul style="color: #2c3e50; line-height: 1.6; text-align: left; max-width: 600px; margin: 0 auto; padding-left: 20px;">
        <li><strong>10 Conversion Tasks:</strong> Full support for all format combinations</li>
        <li><strong>Lab Integration:</strong> 131 HHMI lab paths and 126 project codes</li>
        <li><strong>Smart Mode:</strong> Auto-detect file formats and plan conversions</li>
        <li><strong>Flexible Execution:</strong> Run locally or submit to cluster</li>
        <li><strong>AI Assistant:</strong> Get conversion guidance and parameter suggestions</li>
        <li><strong>Metadata Preservation:</strong> Automatic OME-Zarr metadata handling</li>
        <li><strong>Cost Estimation:</strong> See cluster costs before submitting jobs</li>
        <li><strong>HTTP Support:</strong> Convert from HTTP-served datasets</li>
        </ul>
        </div>
        </div>
        """)
        
        # Let's convert button
        self.start_btn = pn.widgets.Button(
            name="🚀 Let's Convert!",
            button_type="primary",
            width=280,
            height=65,
            styles={
                'font-size': '1.4em',
                'font-weight': '600',
                'margin': '30px auto',
                'display': 'block',
                'border-radius': '16px',
                'background': '#87CEEB !important',
                'border': 'none !important',
                'color': 'white !important',
                'box-shadow': '0 6px 20px rgba(135, 206, 235, 0.4)',
                'cursor': 'pointer'
            }
        )
        # Apply additional styling for gradient
        self.start_btn.stylesheets = ["""
        .bk-btn-primary {
            background: linear-gradient(135deg, #87CEEB 0%, #4682B4 100%) !important;
            border: none !important;
        }
        """]
        self.start_btn.on_click(self.show_conversion_page)
        
        self.welcome_layout = pn.Column(
            welcome_title,
            explanation,
            pn.Row(pn.Spacer(), self.start_btn, pn.Spacer()),
            sizing_mode="stretch_width",
            styles={'max-width': '900px', 'margin': '0 auto', 'padding': '40px 20px'}
        )
        
    def create_conversion_page(self):
        """Create the main conversion configuration page"""
        # Page header with progress steps
        header = pn.pane.Markdown("""
        # 🔬 TensorSwitch Conversion Workflow

        <div style="background: #e3f2fd; padding: 20px; border-radius: 10px; margin-bottom: 30px;">
        <h3 style="margin-top: 0; color: #1976d2;">📋 Simple 3-Step Process:</h3>
        <p style="margin-bottom: 0; font-size: 1.1em;">
        <strong>Step 1:</strong> Enter file paths → <strong>Step 2:</strong> Configure conversion strategy → <strong>Step 3:</strong> Execute conversion
        </p>
        </div>
        """, styles={'text-align': 'center', 'margin-bottom': '20px'})
        
        # STEP 1: File paths section with clear numbering
        step1_header = pn.pane.Markdown("""
        ## 📁 Step 1: Specify Input and Output Paths
        """, styles={'color': '#1976d2', 'margin-bottom': '15px'})

        
        self.input_widget = pn.widgets.TextInput(
            name="📁 Input File/Directory Path", 
            placeholder="/nrs/[lab]/data/file.tif or /groups/[lab]/data/file.nd2",
            width=600,
            styles={'margin-bottom': '10px'}
        )
        self.output_widget = pn.widgets.TextInput(
            name="📤 Output Directory Path", 
            placeholder="/nrs/[lab]/results/output_folder",
            width=600,
            styles={'margin-bottom': '10px'}
        )
        
        # Create simple path helper if lab paths system is available
        self.path_helper_widget = None
        self.path_helper = None
        if LAB_PATHS_AVAILABLE:
            try:
                self.path_helper_widget, self.path_helper = create_simple_path_helper_panel()
                path_selector_widget = self.path_helper_widget
            except Exception as e:
                print(f"Path helper creation failed: {e}")
                path_selector_widget = pn.pane.Markdown(
                    "*Path helper temporarily unavailable*",
                    styles={'margin': '0', 'padding': '5px', 'font-style': 'italic', 'color': '#666'}
                )
        else:
            path_selector_widget = pn.pane.Markdown(
                "*Path helper not loaded*",
                styles={'margin': '0', 'padding': '5px', 'font-style': 'italic', 'color': '#666'}
            )
        
        file_section = pn.Column(
            step1_header,
            path_selector_widget,
            pn.layout.Divider(),
            self.input_widget,
            self.output_widget,
            styles={
                'background': '#ffffff',
                'padding': '20px',
                'border-radius': '10px',
                'margin-bottom': '20px',
                'border': '2px solid #C8E6C9',
                'border-left': '6px solid #81C784',
                'min-width': '100%',
                'overflow': 'visible'
            }
        )
        
        # STEP 2: Workflow mode selection
        step2_header = pn.pane.Markdown("""
        ## 🔄 Step 2: Configure Conversion Strategy
        """, styles={'color': '#1976d2', 'margin-bottom': '15px'})

        mode_section = pn.Column(
            step2_header,
            pn.pane.Markdown("""
            **Choose your preferred workflow:**
            - **🤖 Smart Mode**: Auto-detect input format and choose output (recommended for beginners)
            - **⚙️ Manual Mode**: Traditional task selection for advanced users
            """, styles={'background': '#f8f9fa', 'padding': '15px', 'border-radius': '8px', 'margin-bottom': '15px'}),
            pn.Param(
                self,
                parameters=['workflow_mode'],
                widgets={
                    'workflow_mode': {'type': pn.widgets.RadioButtonGroup, 'name': 'Mode'}
                }
            ),
            styles={
                'background': '#ffffff',
                'padding': '20px',
                'border-radius': '10px',
                'margin-bottom': '10px'
            },
            visible=False  # Initially hidden - shown after Step 1
        )

        # STEP 2A: Format detection section (dynamic - only for smart mode)
        format_header = pn.pane.Markdown("""
        ## 🔍 Step 2A: Smart Format Detection (Smart Mode Only)
        """, styles={'color': '#1976d2', 'margin-bottom': '15px'})
        format_content = pn.pane.Markdown(
            "**🔄 Format Detection**: Enter input path above to auto-detect format and view file metadata",
            styles={'background': '#e3f2fd', 'padding': '15px', 'border-radius': '8px', 'margin-bottom': '20px'}
        )
        self.format_section = pn.Column(format_header, format_content, visible=False)

        # STEP 2B: Output format selection section (smart mode only)
        output_header = pn.pane.Markdown("""
        ## 📤 Step 2B: Configure Output Format (Smart Mode Only)
        """, styles={'color': '#1976d2', 'margin-bottom': '15px'})

        self.output_section = pn.Column(
            output_header,
            pn.pane.Markdown("""
            **Choose your desired output format and processing options:**
            - **Zarr3**: Latest format with sharding support (recommended)
            - **Zarr2**: Legacy format for broader compatibility
            - **N5**: Alternative chunked format
            """, styles={'background': '#f8f9fa', 'padding': '10px', 'border-radius': '5px', 'margin-bottom': '10px'}),
            pn.Param(
                self,
                parameters=['output_format', 'max_downsample_level'],
                widgets={
                    'output_format': {'type': pn.widgets.Select, 'name': 'Output format'},
                    'max_downsample_level': {'type': pn.widgets.IntSlider, 'name': 'Max downsample level (0=no downsampling)'},
                }
            ),
            styles={'background': '#ffffff', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '10px'},
            visible=False
        )

        # Conversion plan section (dynamic)
        plan_header = pn.pane.Markdown("### 🗂️ Conversion Plan")
        plan_content = pn.pane.Markdown(
            "**Conversion Plan**: Configure input and output to see execution plan",
            styles={'background': '#f8f9fa', 'padding': '15px', 'border-radius': '8px', 'margin-bottom': '20px'}
        )

        # Confirmation button for proceeding to Step 3
        self.confirm_plan_btn = pn.widgets.Button(
            name="✅ I like this conversion plan - Next Step!",
            button_type="primary",
            visible=False,
            styles={
                'margin': '20px 0',
                'width': '100%',
                'height': '50px',
                'font-size': '16px',
                'font-weight': '600',
                'border-radius': '12px',
                'background': '#87CEEB !important',
                'border': 'none !important',
                'color': 'white !important',
                'box-shadow': '0 4px 12px rgba(135, 206, 235, 0.4)',
                'cursor': 'pointer'
            }
        )
        # Apply additional styling after creation
        self.confirm_plan_btn.stylesheets = ["""
        .bk-btn-primary {
            background: linear-gradient(135deg, #87CEEB 0%, #4682B4 100%) !important;
            border: none !important;
        }
        """]
        self.confirm_plan_btn.on_click(self._confirm_conversion_plan)

        self.plan_section = pn.Column(
            plan_header,
            plan_content,
            self.confirm_plan_btn,
            visible=False
        )
        
        # Task selection with helpful description
        task_description = pn.pane.Markdown("""
        **Select the conversion task based on your input and desired output formats:**

        **Zarr3 Format (latest, with sharding support):**
        - **tiff_to_zarr3_s0**: Convert TIFF files to Zarr3 with OME metadata
        - **nd2_to_zarr3_s0**: Convert Nikon ND2 files to Zarr3 with OME metadata
        - **ims_to_zarr3_s0**: Convert Imaris IMS files to Zarr3 with OME metadata
        - **downsample_shard_zarr3**: Create downsampled levels from existing Zarr3

        **Zarr2 Format (legacy, broader compatibility):**
        - **tiff_to_zarr2_s0**: Convert TIFF files to Zarr2 with OME metadata
        - **nd2_to_zarr2_s0**: Convert Nikon ND2 files to Zarr2 with OME metadata
        - **ims_to_zarr2_s0**: Convert Imaris IMS files to Zarr2 with OME metadata
        - **downsample_zarr2**: Create downsampled levels from existing Zarr2

        **Other Formats:**
        - **n5_to_zarr2**: Convert N5 format to Zarr2 format
        - **n5_to_n5**: Re-chunk existing N5 files
        """, styles={'background': '#f8f9fa', 'padding': '10px', 'border-radius': '5px', 'margin-bottom': '10px'})
        
        # STEP 2C: Task selection section (manual mode only)
        manual_task_header = pn.pane.Markdown("""
        ## ⚙️ Step 2C: Manual Task Selection (Manual Mode Only)
        """, styles={'color': '#1976d2', 'margin-bottom': '15px'})

        # Confirmation button for manual mode
        self.confirm_manual_btn = pn.widgets.Button(
            name="✅ I've selected the right task - Next Step!",
            button_type="primary",
            visible=False,
            styles={
                'margin': '20px 0',
                'width': '100%',
                'height': '50px',
                'font-size': '16px',
                'font-weight': '600',
                'border-radius': '12px',
                'background': '#87CEEB !important',
                'border': 'none !important',
                'color': 'white !important',
                'box-shadow': '0 4px 12px rgba(135, 206, 235, 0.4)',
                'cursor': 'pointer'
            }
        )
        # Apply additional styling after creation
        self.confirm_manual_btn.stylesheets = ["""
        .bk-btn-primary {
            background: linear-gradient(135deg, #87CEEB 0%, #4682B4 100%) !important;
            border: none !important;
        }
        """]
        self.confirm_manual_btn.on_click(self._confirm_conversion_plan)

        self.task_section = pn.Column(
            manual_task_header,
            task_description,
            pn.Param(
                self,
                parameters=['task'],
                widgets={'task': {'type': pn.widgets.Select, 'name': 'Select conversion task'}}
            ),
            self.confirm_manual_btn,
            styles={'background': '#ffffff', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '10px'},
            visible=False  # Start hidden since we default to smart mode
        )
        
        # Execution location with clear options
        execution_description = pn.pane.Markdown("""
        **Choose where to run your conversion:**
        - **Run locally**: Process on current machine (good for small files, immediate results)
        - **Submit to cluster**: Use LSF cluster (recommended for large files, requires project billing)
        """, styles={'background': '#f8f9fa', 'padding': '10px', 'border-radius': '5px', 'margin-bottom': '10px'})
        
        # STEP 3: Execution configuration
        step3_header = pn.pane.Markdown("""
        ## 🚀 Step 3: Configure Execution & Run
        """, styles={'color': '#1976d2', 'margin-bottom': '15px'})

        execution_section = pn.Column(
            step3_header,
            execution_description,
            pn.Param(
                self,
                parameters=['run_locally'],
                widgets={'run_locally': {'type': pn.widgets.Checkbox, 'name': 'Run locally (uncheck to submit to cluster)'}}
            ),
            styles={'background': '#f8f9fa', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px'},
            visible=False  # Initially hidden - shown after Step 3
        )
        
        # Conditional parameters containers
        self.local_params = pn.Column(
            pn.pane.Markdown("### 🔧 Local Processing Options"),
            pn.Param(
                self,
                parameters=['use_ome_structure', 'use_shard', 'memory_limit'],
                widgets={
                    'use_ome_structure': {'type': pn.widgets.Checkbox, 'name': 'Use OME-Zarr structure'},
                    'use_shard': {'type': pn.widgets.Checkbox, 'name': 'Use sharded format (Zarr3 only)'},
                    'memory_limit': {'type': pn.widgets.IntSlider, 'name': 'Memory limit (%)'},
                }
            ),
            styles={
                'background': '#e8f5e8',
                'padding': '15px',
                'border-radius': '8px',
                'margin-bottom': '10px'
            }
        )

        # Advanced shape parameters section (conditional)
        self.shape_params_section = pn.Column(
            pn.pane.Markdown("**🎛️ Advanced Shape Parameters (Optional):**"),
            pn.pane.Markdown("*Dimension order will be shown here when input file is detected*", name="dimension_order_display"),
            pn.Param(
                self,
                parameters=['custom_shard_shape', 'custom_chunk_shape'],
                widgets={
                    'custom_shard_shape': {'type': pn.widgets.TextInput, 'name': 'Custom shard shape (e.g., 128,576,576)', 'placeholder': 'Leave empty for defaults'},
                    'custom_chunk_shape': {'type': pn.widgets.TextInput, 'name': 'Custom chunk shape (e.g., 32,32,32)', 'placeholder': 'Leave empty for defaults'},
                }
            ),
            visible=False,  # Start hidden, show when sharding is enabled
            styles={'background': '#f0f8f0', 'padding': '15px', 'border-radius': '8px', 'margin-top': '10px'}
        )

        # Add to local params
        self.local_params.append(self.shape_params_section)
        
        # Dask JobQueue description
        dask_description = pn.pane.Markdown("""
        **🚀 Dask JobQueue (Advanced):**
        - Uses advanced task management and error handling
        - Creates independent LSF jobs with local Dask clusters
        - Better CPU/memory utilization than traditional LSF
        - Recommended for large datasets and complex workflows
        """, styles={'background': '#e8f5e8', 'padding': '10px', 'border-radius': '5px', 'margin-bottom': '10px'})

        self.cluster_params = pn.Column(
            pn.pane.Markdown("### 🏢 Cluster Submission Options"),
            pn.Param(
                self,
                parameters=['project', 'cores', 'wall_time', 'num_volumes', 'use_dask_jobqueue'],
                widgets={
                    'project': {'type': pn.widgets.Select, 'name': 'Project for billing'},
                    'cores': {'type': pn.widgets.TextInput, 'name': 'CPU cores'},
                    'wall_time': {'type': pn.widgets.TextInput, 'name': 'Wall time (HH:MM)'},
                    'num_volumes': {'type': pn.widgets.IntSlider, 'name': 'Parallel volumes'},
                    'use_dask_jobqueue': {'type': pn.widgets.Checkbox, 'name': 'Use Dask JobQueue (advanced)'},
                }
            ),
            dask_description,
            pn.pane.Markdown("**Plus all local processing options above**"),
            styles={
                'background': '#fff3cd',
                'padding': '15px',
                'border-radius': '8px',
                'margin-bottom': '10px'
            },
            visible=False
        )
        
        # Action buttons
        self.back_btn = pn.widgets.Button(
            name="← Back to Welcome",
            button_type="light",
            styles={
                'height': '42px',
                'border-radius': '10px',
                'font-weight': '500',
                'border': '2px solid #ddd',
                'background': '#f8f9fa',
                'color': '#495057'
            }
        )
        self.preview_btn = pn.widgets.Button(
            name="🔍 Preview Job",
            button_type="light",
            styles={
                'height': '42px',
                'border-radius': '10px',
                'font-weight': '500',
                'border': '2px solid #ddd',
                'background': '#f8f9fa',
                'color': '#495057'
            }
        )
        self.submit_btn = pn.widgets.Button(
            name="🚀 Run Job",
            button_type="primary",
            styles={
                'height': '42px',
                'font-weight': '600',
                'border-radius': '10px',
                'background': '#87CEEB !important',
                'border': 'none !important',
                'color': 'white !important',
                'box-shadow': '0 3px 10px rgba(135, 206, 235, 0.4)'
            }
        )
        # Apply additional styling
        self.submit_btn.stylesheets = ["""
        .bk-btn-primary {
            background: linear-gradient(135deg, #87CEEB 0%, #4682B4 100%) !important;
            border: none !important;
        }
        """]
        
        self.back_btn.on_click(self.show_welcome_page)
        self.preview_btn.on_click(self.preview_job)
        self.submit_btn.on_click(self.submit_job)
        
        button_row = pn.Row(
            self.back_btn,
            pn.Spacer(),
            self.preview_btn,
            self.submit_btn,
            styles={'margin': '20px 0'},
            visible=False  # Initially hidden - shown after Step 3
        )
        
        # Preview box
        self.preview_box = pn.pane.Markdown(
            "**Job Preview**: Click '🔍 Preview Job' to see job details",
            styles={'background': '#e3f2fd', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #2196f3', 'margin-bottom': '15px'},
            visible=False  # Initially hidden - shown with buttons after Step 3
        )
        
        # Create Step 2 container (combines all Step 2 subsections)
        self.step2_container = pn.Column(
            mode_section,
            self.format_section,
            self.output_section,
            self.plan_section,
            self.task_section,
            styles={
                'border': '2px solid #FFE0B2',
                'border-left': '6px solid #FFB74D',
                'border-radius': '10px',
                'padding': '15px',
                'margin-bottom': '20px'
            },
            visible=False
        )

        # Create Step 3 container (includes header and all Step 3 content)
        self.step3_container = pn.Column(
            execution_section,
            self.local_params,
            self.cluster_params,
            button_row,
            self.preview_box,
            styles={
                'border': '2px solid #BBDEFB',
                'border-left': '6px solid #64B5F6',
                'border-radius': '10px',
                'padding': '15px',
                'margin-bottom': '20px'
            },
            visible=False
        )

        # Store section references for progressive disclosure
        self.file_section = file_section
        self.mode_section = mode_section
        self.execution_section = execution_section
        self.button_row = button_row

        # Main conversion content - now using containers (wider to accommodate path helper)
        conversion_content = pn.Column(
            header,
            file_section,
            self.step2_container,
            self.step3_container,
            sizing_mode="stretch_width",
            styles={'max-width': '900px', 'margin': '0 auto', 'padding': '20px'}
        )

        # Create conversion layout without sidebar (AI will be floating)
        self.conversion_layout = conversion_content

        # Create floating AI assistant if available
        self.floating_ai = None
        if AI_ASSISTANT_AVAILABLE:
            self.floating_ai = create_floating_ai_assistant(
                get_gui_state_callback=self.get_gui_state,
                set_gui_params_callback=self.set_gui_params
            )

        # Initialize plan confirmation state
        self._plan_confirmed = False

        # Set initial visibility - start with progressive disclosure
        self._update_progressive_disclosure()

        # Watch for run_locally changes to show/hide cluster options
        self.param.watch(self.update_execution_options, 'run_locally')
        # Watch for task changes to handle Zarr2-specific logic
        self.param.watch(self.update_task_options, 'task')
        # Watch for use_shard changes to show/hide custom shape options
        self.param.watch(self.update_shape_options_visibility, 'use_shard')
        # Watch for input path changes to trigger format detection AND progressive disclosure
        self.input_widget.param.watch(self.analyze_input_file, 'value')
        self.input_widget.param.watch(self._check_step1_completion, 'value')
        self.output_widget.param.watch(self._check_step1_completion, 'value')
        # Watch for output configuration changes to update conversion plan
        self.param.watch(self.update_conversion_plan, 'output_format')
        self.param.watch(self.update_conversion_plan, 'max_downsample_level')
        # Watch for workflow mode changes to trigger progressive disclosure
        self.param.watch(self._update_workflow_visibility, 'workflow_mode')
        self.param.watch(self._check_step2_completion, 'workflow_mode')
        # Watch for step 2 configuration completion triggers
        self.param.watch(self._check_step2_completion_config, 'output_format')
        self.param.watch(self._check_step2_completion_config, 'task')
        
    def create_progress_page(self):
        """Create the progress monitoring page"""
        # Page header
        progress_header = pn.pane.Markdown("""
        # 🔬 TensorSwitch - Processing
        ---
        """, styles={'text-align': 'center', 'margin-bottom': '30px'})
        
        # Progress bar
        self.progress_bar = pn.widgets.Progress(
            name='Conversion Progress',
            value=0,
            max=100,
            styles={'margin': '20px 0'}
        )
        
        # Progress Status Box
        self.status = pn.pane.Markdown(
            "**Status**: 🟢 Ready to process data",
            styles={'background': '#e8f5e8', 'padding': '20px', 'border-radius': '10px', 'border-left': '4px solid #28a745'}
        )
        
        # Cancel button
        self.cancel_btn = pn.widgets.Button(name="🛑 Cancel Job", button_type="danger", visible=False)
        self.cancel_btn.on_click(self.cancel_job)
        
        # Back to conversion button
        self.back_to_conversion_btn = pn.widgets.Button(name="← Back to Configuration", button_type="light")
        self.back_to_conversion_btn.on_click(self.show_conversion_page)
        
        button_row = pn.Row(
            self.back_to_conversion_btn,
            pn.Spacer(),
            self.cancel_btn,
            styles={'margin': '20px 0'}
        )
        
        self.progress_layout = pn.Column(
            progress_header,
            self.progress_bar,
            self.status,
            button_row,
            sizing_mode="stretch_width",
            styles={'max-width': '800px', 'margin': '0 auto', 'padding': '20px'}
        )
        
    def show_welcome_page(self, event=None):
        """Show the welcome page"""
        self.current_page = "welcome"
        # Clear main container and add welcome content
        self.main_container.clear()
        self.main_container.append(self.welcome_layout)
        
    def show_conversion_page(self, event=None):
        """Show the conversion configuration page with floating AI"""
        self.current_page = "conversion"
        # Clear main container and add conversion content
        self.main_container.clear()
        self.main_container.append(self.conversion_layout)

        # Add floating AI if available (only on conversion page)
        if AI_ASSISTANT_AVAILABLE and hasattr(self, 'floating_ai') and self.floating_ai:
            self.main_container.append(self.floating_ai)
        
    def show_progress_page(self, event=None):
        """Show the progress monitoring page"""
        self.current_page = "progress"
        # Clear main container and add progress content
        self.main_container.clear()
        self.main_container.append(self.progress_layout)
        
    def update_execution_options(self, *args, **kwargs):
        """Update visibility of execution-specific options"""
        if self.run_locally:
            self.cluster_params.visible = False
        else:
            self.cluster_params.visible = True

    def update_task_options(self, *args, **kwargs):
        """Update parameter options based on selected task"""
        # Disable sharding for Zarr2 tasks since it's not supported
        if self.task and 'zarr2' in self.task:
            self.use_shard = False
            # Could disable the widget here if needed

        # Note: Zarr2 tasks automatically ignore use_shard parameter in backend

    def update_shape_options_visibility(self, *args, **kwargs):
        """Update visibility of custom shape options based on sharding setting"""
        # Show shape parameters only when sharding is enabled
        if hasattr(self, 'shape_params_section'):
            self.shape_params_section.visible = self.use_shard

    def get_gui_state(self):
        """Get current GUI state for context-aware AI responses (Phase 1)"""
        state = {
            'input_path': self.input_widget.value if hasattr(self, 'input_widget') else None,
            'output_path': self.output_widget.value if hasattr(self, 'output_widget') else None,
            'workflow_mode': self.workflow_mode if hasattr(self, 'workflow_mode') else None,
            'task': self.task if hasattr(self, 'task') else None,
            'detected_info': self.input_analysis if hasattr(self, 'input_analysis') else None,
            'output_format': self.output_format if hasattr(self, 'output_format') else None,
            'cores': self.cores if hasattr(self, 'cores') else None,
            'memory': self.memory if hasattr(self, 'memory') else None,
            'wall_time': self.wall_time if hasattr(self, 'wall_time') else None,
            'num_volumes': self.num_volumes if hasattr(self, 'num_volumes') else None,
            'use_dask': self.use_dask_jobqueue if hasattr(self, 'use_dask_jobqueue') else None,
            'run_locally': self.run_locally if hasattr(self, 'run_locally') else None,
        }
        return state

    def set_gui_params(self, params):
        """Set GUI parameters from AI suggestions (Phase 2)"""
        # Check if any cluster-specific parameters are being set
        cluster_params = {'cores', 'memory', 'wall_time', 'num_volumes', 'use_dask_jobqueue'}
        has_cluster_params = any(k in params for k in cluster_params)

        # If setting cluster parameters, automatically switch to cluster mode
        if has_cluster_params and hasattr(self, 'run_locally'):
            self.run_locally = False  # Uncheck "Run Locally"

        if 'cores' in params and hasattr(self, 'cores'):
            self.cores = params['cores']
        if 'memory' in params and hasattr(self, 'memory'):
            self.memory = params['memory']
        if 'wall_time' in params and hasattr(self, 'wall_time'):
            self.wall_time = params['wall_time']
        if 'num_volumes' in params and hasattr(self, 'num_volumes'):
            self.num_volumes = params['num_volumes']
        if 'use_dask_jobqueue' in params and hasattr(self, 'use_dask_jobqueue'):
            self.use_dask_jobqueue = params['use_dask_jobqueue']
        if 'output_format' in params and hasattr(self, 'output_format'):
            self.output_format = params['output_format']
        if 'max_downsample_level' in params and hasattr(self, 'max_downsample_level'):
            self.max_downsample_level = params['max_downsample_level']

        # Auto-confirm Step 2 when AI applies parameters
        self._plan_confirmed = True
        if hasattr(self, 'confirm_plan_btn'):
            self.confirm_plan_btn.visible = False
        if hasattr(self, 'confirm_manual_btn'):
            self.confirm_manual_btn.visible = False
        self._update_progressive_disclosure()

    def analyze_input_file(self, *args, **kwargs):
        """Analyze input file when path changes"""
        if not self.format_detector:
            return

        input_path = self.input_widget.value
        if not input_path:
            return

        try:
            # Perform format detection
            self.input_analysis = self.format_detector.analyze_input(input_path)

            # Update format section with analysis results
            if hasattr(self, 'format_section'):
                self._update_format_section()

        except Exception as e:
            print(f"Format analysis failed: {e}")

    def _update_format_section(self):
        """Update the format section with analysis results"""
        if not self.input_analysis or not hasattr(self, 'format_section'):
            return

        if self.input_analysis.get('error'):
            # Show error
            content = f"**Analysis Result**: ❌ {self.input_analysis['error']}"
            style = {'background': '#f8d7da', 'padding': '15px', 'border-radius': '8px'}
        elif self.input_analysis.get('format'):
            # Show detected format and metadata
            summary = self.format_detector.format_summary(self.input_analysis)
            content = f"**Auto-Detected Input**:  \n{summary}"  # Added two spaces before newline for proper markdown break
            style = {'background': '#d4edda', 'padding': '15px', 'border-radius': '8px'}

            # Update dimension order display in shape parameters
            self._update_dimension_order_display()
        else:
            content = "**🔄 Format Conversion**: Enter input path to auto-detect format"
            style = {'background': '#e3f2fd', 'padding': '15px', 'border-radius': '8px'}

        # Update format section content
        self.format_section[1].object = content
        self.format_section[1].styles = style

        # Update output format options based on detected input format
        self._update_output_format_options()

        # Update conversion plan
        self.update_conversion_plan()

    def _update_dimension_order_display(self):
        """Update the dimension order display in the shape parameters section"""
        if not self.input_analysis or not hasattr(self, 'shape_params_section'):
            return

        shape = self.input_analysis.get('shape')
        if shape:
            dimension_order = self.format_detector._infer_dimension_order(shape)
            if dimension_order:
                dim_text = f"**📐 Detected dimension order**: {' × '.join(dimension_order)}\n"
                dim_text += f"**Shape**: {shape}\n\n"
                dim_text += "*Use this order when specifying custom shapes (e.g., for CZYX use: C,Z,Y,X)*"

                # Update the dimension order display markdown
                self.shape_params_section[1].object = dim_text
            else:
                self.shape_params_section[1].object = "*Dimension order will be shown here when input file is detected*"
        else:
            self.shape_params_section[1].object = "*Dimension order will be shown here when input file is detected*"

    def _update_output_format_options(self):
        """Update output format dropdown based on detected input format"""
        if not self.input_analysis or not self.task_planner:
            return

        input_format = self.input_analysis.get('format')
        if input_format:
            compatible_outputs = self.task_planner.get_compatible_outputs(input_format)
            if compatible_outputs:
                # Update parameter choices
                self.param.output_format.objects = compatible_outputs
                # Set default to first compatible format if current selection is not compatible
                if self.output_format not in compatible_outputs:
                    self.output_format = compatible_outputs[0]

    def update_conversion_plan(self, *args, **kwargs):
        """Update conversion plan when input/output configuration changes"""
        if not self.input_analysis or not self.task_planner:
            return

        input_format = self.input_analysis.get('format')
        if not input_format:
            return

        try:
            # Create downsample levels list
            downsample_levels = list(range(self.max_downsample_level + 1))

            # Generate conversion plan
            self.conversion_plan = self.task_planner.create_conversion_plan(
                input_format,
                self.output_format,
                downsample_levels
            )

            # Update plan section
            self._update_plan_section()

            # Auto-select the task for the preview
            if self.conversion_plan and not self.conversion_plan.get('error'):
                tasks = self.conversion_plan.get('tasks', [])
                if tasks:
                    # Use the primary conversion task
                    primary_task = tasks[0]['task']
                    if primary_task in self.param.task.objects:
                        self.task = primary_task

        except Exception as e:
            print(f"Conversion plan update failed: {e}")

    def _update_plan_section(self):
        """Update the conversion plan section"""
        if not self.conversion_plan or not hasattr(self, 'plan_section'):
            return

        if self.conversion_plan.get('error'):
            # Show error
            content = f"**Plan Error**: ❌ {self.conversion_plan['error']}"
            style = {'background': '#f8d7da', 'padding': '15px', 'border-radius': '8px'}
            # Hide confirmation button on error
            if hasattr(self, 'confirm_plan_btn'):
                self.confirm_plan_btn.visible = False
        elif self.conversion_plan.get('tasks'):
            # Show conversion plan
            file_size_mb = self.input_analysis.get('size_mb', 0)
            plan_summary = self.task_planner.format_plan_summary(self.conversion_plan, file_size_mb)
            content = f"**Execution Plan**:\n{plan_summary}"
            style = {'background': '#d4edda', 'padding': '15px', 'border-radius': '8px'}
            # Show confirmation button when plan is ready
            if hasattr(self, 'confirm_plan_btn') and not getattr(self, '_plan_confirmed', False):
                self.confirm_plan_btn.visible = True
        else:
            content = "**Conversion Plan**: Configure input and output to see execution plan"
            style = {'background': '#f8f9fa', 'padding': '15px', 'border-radius': '8px'}
            # Hide confirmation button when no plan
            if hasattr(self, 'confirm_plan_btn'):
                self.confirm_plan_btn.visible = False

        # Update plan section content
        self.plan_section[1].object = content
        self.plan_section[1].styles = style

    def _update_progressive_disclosure(self):
        """Update visibility based on progressive step completion"""
        # Step 1: Always visible (input/output paths)
        self.file_section.visible = True

        # Step 2: Show entire container after Step 1 is complete
        step1_complete = self._is_step1_complete()
        self.step2_container.visible = step1_complete

        # Control subsections within Step 2
        if step1_complete:
            self.mode_section.visible = True
            self.format_section.visible = self.workflow_mode == "smart"
            self.output_section.visible = self.workflow_mode == "smart"
            self.plan_section.visible = self.workflow_mode == "smart"
            self.task_section.visible = self.workflow_mode == "manual"
        else:
            # Hide all Step 2 subsections if Step 1 not complete
            self.mode_section.visible = False
            self.format_section.visible = False
            self.output_section.visible = False
            self.plan_section.visible = False
            self.task_section.visible = False

        # Show confirmation buttons when Step 2 configuration is complete
        step2_config_complete = step1_complete and self._is_step2_complete()
        if hasattr(self, 'confirm_plan_btn'):
            # Smart mode confirmation button (shown via _update_plan_section when plan is ready)
            pass
        if hasattr(self, 'confirm_manual_btn'):
            # Manual mode confirmation button
            self.confirm_manual_btn.visible = (step2_config_complete and
                                               self.workflow_mode == "manual" and
                                               not getattr(self, '_plan_confirmed', False))

        # Step 3: Show entire container only after plan is confirmed
        step3_should_show = getattr(self, '_plan_confirmed', False)
        self.step3_container.visible = step3_should_show

        # Control subsections within Step 3
        if step3_should_show:
            self.execution_section.visible = True
            self.local_params.visible = self.run_locally
            self.cluster_params.visible = not self.run_locally
            self.button_row.visible = True
            self.preview_box.visible = True
        else:
            # Hide all Step 3 subsections if not confirmed
            self.execution_section.visible = False
            self.local_params.visible = False
            self.cluster_params.visible = False
            self.button_row.visible = False
            self.preview_box.visible = False

    def _is_step1_complete(self):
        """Check if Step 1 (paths) is complete"""
        input_path = self.input_widget.value.strip() if self.input_widget.value else ""
        output_path = self.output_widget.value.strip() if self.output_widget.value else ""
        return len(input_path) > 0 and len(output_path) > 0


    def _is_step2_complete(self):
        """Check if Step 2 (configuration) is complete"""
        if self.workflow_mode == "smart":
            # Smart mode: need output format selected
            return bool(self.output_format)
        else:
            # Manual mode: need task selected
            return bool(self.task)

    def _confirm_conversion_plan(self, event):
        """Handle confirmation of conversion plan"""
        self._plan_confirmed = True
        self.confirm_plan_btn.visible = False
        self._update_progressive_disclosure()

    def _check_step1_completion(self, *args, **kwargs):
        """Check Step 1 completion and update disclosure"""
        self._update_progressive_disclosure()

    def _check_step2_completion(self, *args, **kwargs):
        """Check Step 2 completion and update disclosure"""
        self._update_progressive_disclosure()

    def _check_step2_completion_config(self, *args, **kwargs):
        """Check Step 2 configuration completion and update disclosure"""
        self._update_progressive_disclosure()

    def _update_workflow_visibility(self, *args, **kwargs):
        """Update section visibility based on workflow mode"""
        # Use progressive disclosure instead of just workflow mode
        self._update_progressive_disclosure()

    def sync_path_selector_to_form(self):
        """Sync path helper selections to form fields"""
        # Path helper shows suggestions for users to copy-paste
        pass
    
    def update_progress_display(self):
        """Periodic callback to update GUI with latest progress data"""
        if not self.job_running:
            return
            
        total = self.progress_data['total_chunks']
        processed = self.progress_data['processed_chunks']
        stage = self.progress_data['stage']
        
        if total > 0:
            if stage == "Writing metadata":
                self.progress_bar.value = 98
                self.status.object = f"**Status**: 🟡 Finalizing...\n\n📊 **Progress**: Nearly complete\n\n🔄 Stage: {stage}"
            elif processed > 0:
                progress_percent = min(95, int((processed / total) * 100))
                self.progress_bar.value = progress_percent
                self.status.object = f"**Status**: 🟡 Processing chunks...\n\n📊 **Progress**: {processed:,} / {total:,} chunks ({progress_percent}%)\n\n🔄 Stage: {stage}"
            elif stage == "Dataset loaded":
                self.status.object = f"**Status**: 🟡 Dataset analysis complete\n\n📊 **Total chunks**: {total:,}\n\n🔄 Stage: {stage}"
    
    def preview_job(self, event):
        """Preview what job would be submitted"""
        # Sync path selector to form fields first
        self.sync_path_selector_to_form()
        
        input_path = self.input_widget.value
        output_path = self.output_widget.value
        
        if not input_path or not output_path:
            self.preview_box.object = "**Job Preview**: ⚠️ Please enter both input and output paths"
            self.preview_box.styles = {'background': '#fff3cd', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #ffc107', 'margin-bottom': '15px'}
            return
            
        # Only require project selection for cluster jobs, not local runs
        if not self.run_locally and not self.project:
            self.preview_box.object = "**Job Preview**: ⚠️ Please select a project for billing (required for cluster jobs)"
            self.preview_box.styles = {'background': '#fff3cd', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #ffc107', 'margin-bottom': '15px'}
            return
        
        execution_mode = "locally" if self.run_locally else "on LSF cluster"

        # Add notes about format and advanced options
        format_note = ""
        if 'zarr2' in self.task:
            format_note = " (Zarr2 format)"
        elif 'zarr3' in self.task:
            format_note = " (Zarr3 format)"

        sharding_note = ""
        if 'zarr2' in self.task and self.use_shard:
            sharding_note = " (disabled for Zarr2)"

        dask_note = ""
        if not self.run_locally and self.use_dask_jobqueue:
            dask_note = " with Dask JobQueue"

        # Determine level to use (smart mode uses max_downsample_level, manual mode would use level if it existed)
        level_to_use = self.max_downsample_level if self.workflow_mode == "smart" else 0

        # Add custom shape info
        custom_shapes_info = ""
        if self.custom_shard_shape and self.custom_shard_shape.strip():
            custom_shapes_info += f"\n- **Custom Shard Shape**: {self.custom_shard_shape.strip()}"
        if self.custom_chunk_shape and self.custom_chunk_shape.strip():
            custom_shapes_info += f"\n- **Custom Chunk Shape**: {self.custom_chunk_shape.strip()}"

        # Add cost estimation if available
        cost_info = ""
        if COST_ESTIMATOR_AVAILABLE and not self.run_locally:
            try:
                # Parse wall time (format: HH:MM or MM)
                wall_time_str = self.wall_time.strip()
                if ':' in wall_time_str:
                    hours, minutes = wall_time_str.split(':')
                    wall_time_hours = int(hours) + int(minutes) / 60
                else:
                    wall_time_hours = int(wall_time_str) / 60  # Assume minutes

                num_cores = int(self.cores)
                num_volumes = int(self.num_volumes)
                total_slots = num_cores * num_volumes  # Account for parallel jobs

                # Try to get chunk information from input analysis
                if hasattr(self, 'input_analysis') and self.input_analysis:
                    shape = self.input_analysis.get('shape')
                    # Get chunk shape from custom or defaults
                    chunk_shape_str = self.custom_chunk_shape.strip() if self.custom_chunk_shape else "64,64,64"

                    if shape and chunk_shape_str:
                        try:
                            chunk_shape = tuple(int(x) for x in chunk_shape_str.split(','))
                            total_chunks = calculate_total_chunks(shape, chunk_shape)

                            # Calculate time per volume (chunks split across volumes)
                            chunks_per_volume = total_chunks / num_volumes
                            est_time = estimate_processing_time(int(chunks_per_volume), num_cores)

                            # Calculate costs using total slots (cores × volumes)
                            est_cost = estimate_cluster_cost(total_slots, est_time)
                            max_cost = estimate_cluster_cost(total_slots, wall_time_hours)
                            ai_cost = get_ai_cost()

                            cost_info = f"""
---

### 💰 Cost & Time Estimate

- **Processing Time**: ~{format_time(est_time)} (per volume, {num_volumes} parallel jobs)
- **Total Chunks**: {total_chunks:,} ({int(chunks_per_volume):,} per volume)
- **Total Slots Used**: {total_slots} ({num_cores} cores × {num_volumes} volumes)
- **Cluster Cost**: ${est_cost:.4f} - ${max_cost:.4f} (Est - Max at {wall_time_hours:.1f}h wall time)
- **AI Assistant**: ${ai_cost:.2f}/session
- **TOTAL COST**: ${est_cost + ai_cost:.4f} - ${max_cost + ai_cost:.4f}

---
"""
                        except:
                            pass  # Skip cost estimation if parsing fails
            except:
                pass  # Skip cost estimation if any error occurs

        preview_text = f"""
**Job Preview**: 🔍

- **Task**: `{self.task}`{format_note}
- **Input**: `{input_path}`
- **Output**: `{output_path}`
- **Project**: `{self.project}`
- **Execution**: Run {execution_mode}{dask_note}
- **Max Downsample Level**: {level_to_use}
- **Use Sharding**: {self.use_shard}{sharding_note}
- **OME Structure**: {self.use_ome_structure} {'(auto multiscale metadata)' if self.use_ome_structure else ''}
- **CPU Cores**: {self.cores}
- **Wall Time**: {self.wall_time}
- **Parallel Volumes**: {self.num_volumes}
- **Memory Limit**: {self.memory_limit}%{custom_shapes_info}
{cost_info}
Click **🚀 Run Job** to execute this configuration.
        """

        self.preview_box.object = preview_text
        self.preview_box.styles = {'background': '#e3f2fd', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #2196f3', 'margin-bottom': '15px'}

    def submit_job(self, event):
        """Submit job for execution"""
        input_path = self.input_widget.value
        output_path = self.output_widget.value
        
        if not input_path or not output_path:
            self.preview_box.object = "**Job Preview**: ❌ Please enter both input and output paths"
            self.preview_box.styles = {'background': '#f8d7da', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #dc3545', 'margin-bottom': '15px'}
            return
            
        if not self.project and not self.run_locally:
            self.preview_box.object = "**Job Preview**: ❌ Please select a project for cluster billing"
            self.preview_box.styles = {'background': '#f8d7da', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #dc3545', 'margin-bottom': '15px'}
            return
        
        if self.run_locally:
            # Switch to progress page for local jobs (we can monitor progress)
            self.show_progress_page()
            self.run_local_job(input_path, output_path)
        else:
            # Stay on conversion page for cluster jobs (can't monitor LSF jobs in real-time)
            self.submit_cluster_job(input_path, output_path)
    
    def run_local_job(self, input_path, output_path):
        """Run job locally with progress monitoring"""
        if self.job_running:
            self.status.object = "**Status**: ⚠️ Another job is already running"
            return
            
        # Start job in separate thread
        self.job_running = True
        self.cancel_btn.visible = True
        self.submit_btn.disabled = True
        self.progress_bar.value = 0
        
        # Update preview box to show job has started
        self.preview_box.object = f"**Job Started**: 🚀 Running {self.task}\n\nInput: `{input_path}`\nOutput: `{output_path}`"
        self.preview_box.styles = {'background': '#d4edda', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #28a745', 'margin-bottom': '15px'}
        
        thread = threading.Thread(target=self._execute_local_job, args=(input_path, output_path))
        thread.daemon = True
        thread.start()
    
    def _execute_local_job(self, input_path, output_path):
        """Execute local job with real-time progress monitoring - handles multiscale"""
        try:
            # Determine max level to use
            max_level = self.max_downsample_level if self.workflow_mode == "smart" else 0

            # STEP 1: Run s0 conversion first
            self.status.object = f"**Status**: 🟡 Starting s0 conversion (level 0/{max_level})...\n\n📊 Analyzing dataset..."
            self.status.styles = {'background': '#fff3cd', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #ffc107'}

            s0_success = self._execute_single_level_local(input_path, output_path, level=0, is_downsample=False)

            if not s0_success:
                self._job_finished(success=False, message="s0 conversion failed")
                return

            # STEP 2: Run downsampling for levels 1 through max_level
            if max_level > 0:
                # Determine downsampling task
                if "zarr3" in self.task or "shard" in self.task:
                    downsample_task = "downsample_shard_zarr3"
                elif "zarr2" in self.task:
                    downsample_task = "downsample_zarr2"
                else:
                    downsample_task = None

                if downsample_task:
                    for level in range(1, max_level + 1):
                        self.status.object = f"**Status**: 🟡 Downsampling level {level}/{max_level}...\n\n📊 Processing..."
                        self.status.styles = {'background': '#fff3cd', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #ffc107'}

                        # Adjust base_path for downsampling
                        if self.use_ome_structure:
                            prev_level_path = os.path.join(output_path, f"multiscale/s{level-1}")
                        else:
                            prev_level_path = os.path.join(output_path, f"s{level-1}")

                        level_success = self._execute_single_level_local(
                            prev_level_path, output_path, level=level,
                            is_downsample=True, downsample_task=downsample_task
                        )

                        if not level_success:
                            self._job_finished(success=False, message=f"Downsampling failed at level {level}")
                            return

            # All levels completed successfully
            self._job_finished(success=True, message=f"All {max_level + 1} levels completed successfully! (s0 through s{max_level})")

        except Exception as e:
            self._job_finished(success=False, message=f"Error starting job: {str(e)}")

    def _execute_single_level_local(self, input_path, output_path, level, is_downsample=False, downsample_task=None):
        """Execute a single level locally and monitor progress - returns True on success"""
        try:
            # Determine which task to use
            task = downsample_task if is_downsample else self.task

            cmd = [
                "python", "-u",  # -u for unbuffered output (real-time progress)
                "-m", "tensorswitch",
                "--task", task,
                "--base_path", input_path,
                "--output_path", output_path,
                "--level", str(level),
                "--use_shard", "1" if self.use_shard else "0",
                "--use_ome_structure", "1" if self.use_ome_structure else "0",
                "--memory_limit", str(self.memory_limit),
                "--num_volumes", str(self.num_volumes)
            ]

            # Add custom shape parameters if provided (only for s0, not downsampling)
            if not is_downsample:
                if self.custom_shard_shape and self.custom_shard_shape.strip():
                    cmd.extend(["--custom_shard_shape", self.custom_shard_shape.strip()])
                if self.custom_chunk_shape and self.custom_chunk_shape.strip():
                    cmd.extend(["--custom_chunk_shape", self.custom_chunk_shape.strip()])

            # Execute command with real-time output capture
            # Set PYTHONUNBUFFERED environment variable for immediate output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )

            # Monitor progress in real-time
            self._monitor_job_progress()

            # Check if job completed successfully
            # Note: _monitor_job_progress sets self.current_process = None, so check progress_data
            if hasattr(self, 'last_job_success'):
                return self.last_job_success
            else:
                return False

        except Exception as e:
            print(f"Error executing level {level}: {str(e)}")
            return False
    
    def _monitor_job_progress(self):
        """Monitor job progress by parsing stdout"""
        total_chunks = None
        processed_chunks = 0
        current_stage = "Initializing"
        
        try:
            while True:
                if self.current_process is None:
                    break
                    
                line = self.current_process.stdout.readline()
                if not line:
                    # Process finished
                    return_code = self.current_process.poll()
                    if return_code is not None:
                        if return_code == 0:
                            self._job_finished(success=True, message=f"Conversion completed successfully! Output saved to: `{self.output_widget.value}`")
                        else:
                            self._job_finished(success=False, message="Job failed - check error details above")
                        break
                    continue
                
                # Parse progress from output lines
                line = line.strip()
                
                # Extract total chunks
                if "Total chunks:" in line:
                    try:
                        total_chunks = int(re.search(r'Total chunks: (\d+)', line).group(1))
                        current_stage = "Processing chunks"
                        self.progress_data['total_chunks'] = total_chunks
                        self.progress_data['stage'] = current_stage
                    except:
                        pass
                
                # Extract chunk progress  
                elif "Queued" in line and "chunk writes" in line:
                    try:
                        processed_chunks = int(re.search(r'Queued (\d+)', line).group(1))
                        self.progress_data['processed_chunks'] = processed_chunks
                        self.progress_data['stage'] = current_stage
                    except:
                        pass
                
                # Extract dataset info
                elif "volume shape:" in line:
                    current_stage = "Dataset loaded"
                    self.progress_data['stage'] = current_stage
                
                elif "Writing OME-Zarr metadata" in line:
                    current_stage = "Writing metadata"
                    self.progress_data['stage'] = current_stage
                
                elif "Updating OME metadata" in line:
                    current_stage = "Updating multiscale metadata"
                    self.progress_data['stage'] = current_stage
                
                elif "OME metadata updated successfully" in line:
                    current_stage = "Metadata update complete"
                    self.progress_data['stage'] = current_stage
                
                # Handle errors
                elif "Error" in line or "error" in line or "Failed" in line:
                    self.status.object += f"\n\n❌ **Error**: {line}"
                    
                time.sleep(0.1)  # Small delay to prevent overwhelming the GUI
                
        except Exception as e:
            self._job_finished(success=False, message=f"Error monitoring progress: {str(e)}")
    
    def _job_finished(self, success=True, message=""):
        """Clean up after job completion"""
        self.job_running = False
        self.cancel_btn.visible = False
        self.submit_btn.disabled = False
        self.current_process = None
        self.last_job_success = success  # Store for _execute_single_level_local to check

        if success:
            self.progress_bar.value = 100
            self.status.object = f"**Status**: ✅ {message}"
            self.status.styles = {'background': '#d4edda', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #28a745'}
            self.preview_box.object = "**Job Completed**: ✅ Conversion finished successfully!\n\nClick '🔍 Preview Job' to configure a new job."
            self.preview_box.styles = {'background': '#d4edda', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #28a745', 'margin-bottom': '15px'}
        else:
            self.progress_bar.value = 0
            self.status.object = f"**Status**: ❌ {message}"
            self.status.styles = {'background': '#f8d7da', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #dc3545'}
            self.preview_box.object = "**Job Failed**: ❌ Conversion failed!\n\nCheck the status box below for error details. Click '🔍 Preview Job' to try again."
            self.preview_box.styles = {'background': '#f8d7da', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #dc3545', 'margin-bottom': '15px'}
    
    def cancel_job(self, event):
        """Cancel the running job"""
        if self.current_process:
            self.current_process.terminate()
            self.current_process = None
        self._job_finished(success=False, message="Job cancelled by user")
    
    def _generate_coordinator_script(self, input_path, output_path, max_level):
        """Generate bash coordinator script for sequential multiscale job submission"""

        # Determine downsampling task based on current task
        if "zarr3" in self.task or "shard" in self.task:
            downsample_task = "downsample_shard_zarr3"
        elif "zarr2" in self.task:
            downsample_task = "downsample_zarr2"
        else:
            downsample_task = None

        # Get absolute path to python and tensorswitch
        # IMPORTANT: Use shared filesystem paths accessible from compute nodes
        # The VM's /opt/tensorswitch is LOCAL to the VM and NOT accessible to compute nodes
        # Instead, use the shared installation in the user's home directory on /groups
        tensorswitch_dir = "/groups/scicompsoft/home/chend/temp/downsample_script/tensorswitch/src"
        python_path = "/groups/scicompsoft/home/chend/temp/downsample_script/tensorswitch/.pixi/envs/default/bin/python3.12"

        # Build base command arguments
        base_args = f"--use_shard {'1' if self.use_shard else '0'} --use_ome_structure {'1' if self.use_ome_structure else '0'} --memory_limit {self.memory_limit} --cores {self.cores} --wall_time {self.wall_time} --num_volumes {self.num_volumes} --project {self.project}"

        # Add custom shape parameters if provided
        if self.custom_shard_shape and self.custom_shard_shape.strip():
            base_args += f" --custom_shard_shape {self.custom_shard_shape.strip()}"
        if self.custom_chunk_shape and self.custom_chunk_shape.strip():
            base_args += f" --custom_chunk_shape {self.custom_chunk_shape.strip()}"

        script = f"""#!/bin/bash
# TensorSwitch Multiscale Coordinator Script
# Generated by GUI for sequential level submission
# Levels: s0 through s{max_level}

set -e  # Exit on any error

echo "========================================="
echo "TensorSwitch Multiscale Coordinator"
echo "========================================="
echo "Input: {input_path}"
echo "Output: {output_path}"
echo "Levels: s0 through s{max_level}"
echo "Started: $(date)"
echo "========================================="

cd {tensorswitch_dir}

# Function to extract job IDs from bsub output
extract_job_ids() {{
    grep -oP 'Job <\\K[0-9]+(?=>)' || true
}}

# Function to submit level and wait for completion
submit_and_wait() {{
    local level=$1
    local task=$2
    local base_path=$3
    local output_path=$4

    echo ""
    echo ">>> Submitting level s$level (task: $task)..."

    # Submit jobs and capture output
    output=$({python_path} -m tensorswitch \\
        --task $task \\
        --base_path "$base_path" \\
        --output_path "$output_path" \\
        --level $level \\
        {base_args} \\
        --submit 2>&1)

    echo "$output"

    # Extract job IDs
    job_ids=$(echo "$output" | extract_job_ids | tr '\\n' ' ')

    if [ -z "$job_ids" ]; then
        echo "ERROR: No job IDs found for level s$level"
        exit 1
    fi

    echo ">>> Level s$level jobs submitted: $job_ids"
    echo ">>> Waiting for level s$level to complete..."

    # Build wait condition: ended(job1) && ended(job2) && ...
    wait_condition=""
    for job_id in $job_ids; do
        if [ -z "$wait_condition" ]; then
            wait_condition="ended($job_id)"
        else
            wait_condition="$wait_condition && ended($job_id)"
        fi
    done

    # Wait for all jobs to finish
    bwait -w "$wait_condition" 2>&1 || true

    echo ">>> Level s$level completed at $(date)"
}}

# Submit s0 (initial conversion)
echo ""
echo "=== LEVEL 0: Initial Conversion ==="
submit_and_wait 0 "{self.task}" "{input_path}" "{output_path}"

"""

        # Add downsampling levels if needed
        if max_level > 0 and downsample_task:
            for level in range(1, max_level + 1):
                if self.use_ome_structure:
                    prev_level_path = os.path.join(output_path, f"multiscale/s{level-1}")
                else:
                    prev_level_path = os.path.join(output_path, f"s{level-1}")

                script += f"""
# Submit s{level} (downsample from s{level-1})
echo ""
echo "=== LEVEL {level}: Downsampling ==="
submit_and_wait {level} "{downsample_task}" "{prev_level_path}" "{output_path}"

"""

        script += f"""
echo ""
echo "========================================="
echo "✅ All levels completed successfully!"
echo "Finished: $(date)"
echo "Output location: {output_path}"
echo "========================================="
"""

        return script

    def submit_cluster_job(self, input_path, output_path):
        """Submit job to LSF cluster - uses coordinator for multiscale"""
        # Update preview box to show submission is starting
        self.preview_box.object = f"**Job Submission**: 🚀 Submitting to LSF cluster...\n\nInput: `{input_path}`\nOutput: `{output_path}`\nProject: `{self.project}`"
        self.preview_box.styles = {'background': '#fff3cd', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #ffc107', 'margin-bottom': '15px'}

        try:
            # Determine max level to use (smart mode uses max_downsample_level, manual mode uses 0)
            max_level = self.max_downsample_level if self.workflow_mode == "smart" else 0

            # If max_level > 0, use coordinator script approach
            if max_level > 0:
                self._submit_coordinator_job(input_path, output_path, max_level)
            else:
                # For single level (s0 only), use direct submission
                self._submit_direct_job(input_path, output_path)

        except Exception as e:
            self.preview_box.object = f"**❌ Error Submitting Cluster Job**\n\n{str(e)}"
            self.preview_box.styles = {'background': '#f8d7da', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #dc3545', 'margin-bottom': '15px'}

    def _submit_direct_job(self, input_path, output_path):
        """Submit single level job directly (no coordinator needed)"""
        result = self._submit_single_level_job(input_path, output_path, level=0, is_downsample=False)

        if result and result['job_ids']:
            job_info = f"\n**📋 Job Tracking:**\n"
            for i, (job_id, job_name) in enumerate(zip(result['job_ids'], result['job_names'])):
                job_info += f"- **Job {i+1}**: `{job_name}` (ID: {job_id})\n"
            job_info += f"\n**🔍 Check Status:** `bjobs {' '.join(result['job_ids'])}`\n"
            job_info += f"**📊 Monitor:** `bjobs -w`\n"

            self.preview_box.object = f"""**✅ Cluster Jobs Submitted Successfully!**

**Input:** `{input_path}`
**Output:** `{output_path}`
**Project:** `{self.project}`
**Resources:** {self.cores} cores, {self.wall_time} wall time

**Job Info:** [START]{job_info} [END]

**💡 Tip:** Jobs are running in the background. Check output directory when complete!"""

            self.preview_box.styles = {'background': '#d4edda', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #28a745', 'margin-bottom': '15px'}
        else:
            self.preview_box.object = f"**❌ Cluster Submission Failed!**\n\nNo jobs were submitted successfully."
            self.preview_box.styles = {'background': '#f8d7da', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #dc3545', 'margin-bottom': '15px'}

    def _submit_coordinator_job(self, input_path, output_path, max_level):
        """Submit coordinator job for multiscale processing"""
        import tempfile
        import time

        # Generate coordinator script
        script_content = self._generate_coordinator_script(input_path, output_path, max_level)

        # Write script to persistent location (not /tmp which gets cleaned up)
        # Use user's output directory so it's on shared filesystem accessible to compute nodes
        # output_path is the zarr file/directory, so get its parent directory
        output_dir = os.path.dirname(output_path) if output_path else os.path.expanduser("~/tensorswitch_jobs")
        os.makedirs(output_dir, exist_ok=True)

        # Create script with timestamp to avoid collisions
        timestamp = int(time.time())
        script_path = os.path.join(output_dir, f"coordinator_s0_s{max_level}_{timestamp}.sh")

        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make script executable
        os.chmod(script_path, 0o755)

        # Submit coordinator as LSF job
        job_name = f"tensorswitch_coordinator_s0_s{max_level}"

        cmd = [
            "bsub",
            "-J", job_name,
            "-n", "1",  # Coordinator only needs 1 core
            "-W", "24:00",  # Give plenty of time for all levels
            "-P", self.project,
            "-o", f"{output_dir}/coordinator_{job_name}_%J.log",
            "-e", f"{output_dir}/coordinator_{job_name}_%J.err",
            script_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Extract coordinator job ID
            import re
            match = re.search(r'Job <(\d+)>', result.stdout)
            if match:
                coordinator_job_id = match.group(1)

                self.preview_box.object = f"""**✅ Coordinator Job Submitted Successfully!**

**Input:** `{input_path}`
**Output:** `{output_path}`
**Project:** `{self.project}`
**Levels:** s0 through s{max_level}

**📋 Coordinator Job:**
- **Job Name**: `{job_name}`
- **Job ID**: `{coordinator_job_id}`
- **Script**: `{script_path}`

**What happens next:**
1. Coordinator submits s0 jobs (8 parallel workers)
2. Waits for s0 to complete
3. Submits s1 jobs (8 parallel workers)
4. Waits for s1 to complete
5. Continues through s{max_level}...

**🔍 Monitor Progress:**
- Check coordinator: `bjobs {coordinator_job_id}`
- View coordinator log: `tail -f {output_dir}/coordinator_{job_name}_{coordinator_job_id}.log`
- Check all jobs: `bjobs -w`

**📁 Logs:** `{output_dir}/`

**💡 Tip:** The coordinator orchestrates everything. You can close the GUI - jobs will continue running!"""

                self.preview_box.styles = {'background': '#d4edda', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #28a745', 'margin-bottom': '15px'}
            else:
                raise Exception("Could not extract coordinator job ID from bsub output")
        else:
            self.preview_box.object = f"**❌ Coordinator Submission Failed!**\n\n**Error:**\n```\n{result.stderr[:800]}\n```"
            self.preview_box.styles = {'background': '#f8d7da', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #dc3545', 'margin-bottom': '15px'}

    def _submit_single_level_job(self, input_path, output_path, level, is_downsample=False, downsample_task=None):
        """Submit a single level job (either s0 conversion or downsampling)"""
        try:
            # Determine which task to use
            if is_downsample:
                task = downsample_task
            else:
                task = self.task

            # Build command for cluster submission
            cmd = [
                "python", "-m", "tensorswitch",
                "--task", task,
                "--base_path", input_path,
                "--output_path", output_path,
                "--level", str(level),
                "--use_shard", "1" if self.use_shard else "0",
                "--use_ome_structure", "1" if self.use_ome_structure else "0",
                "--memory_limit", str(self.memory_limit),
                "--cores", str(self.cores),
                "--wall_time", str(self.wall_time),
                "--num_volumes", str(self.num_volumes),
                "--project", self.project,
                "--submit"  # This triggers cluster submission
            ]

            # Add custom shape parameters if provided (only for s0 conversion, not downsampling)
            if not is_downsample:
                if self.custom_shard_shape and self.custom_shard_shape.strip():
                    cmd.extend(["--custom_shard_shape", self.custom_shard_shape.strip()])
                if self.custom_chunk_shape and self.custom_chunk_shape.strip():
                    cmd.extend(["--custom_chunk_shape", self.custom_chunk_shape.strip()])

            # Add Dask JobQueue flag if enabled
            if self.use_dask_jobqueue:
                cmd.append("--use_dask_jobqueue")

            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

            if result.returncode == 0:
                # Extract job IDs from output
                import re
                job_ids = re.findall(r'Job <(\d+)>', result.stdout)
                job_names = re.findall(r'Submitted (\w+)', result.stdout)

                # Return results dict for the main function to collect
                return {
                    'success': True,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'job_ids': job_ids,
                    'job_names': job_names
                }
            else:
                # Job submission failed - log error but don't stop other levels
                print(f"Job submission failed for level {level}: {result.stderr[:500]}")
                return None

        except Exception as e:
            print(f"Error submitting level {level} job: {str(e)}")
            return None

def create_simple_app():
    """Create simple test app"""
    pn.extension()

    # Create a function that returns a fresh GUI instance for each session
    def create_gui_for_session():
        gui = SimpleTensorSwitchGUI()
        # Force show welcome page for each new session
        gui.show_welcome_page()
        return gui.layout

    # Return the function so Panel creates a new instance per session
    return create_gui_for_session

if __name__ == "__main__":
    print("Starting simple TensorSwitch GUI test...")
    print("JupyterHub Environment Detected")
    
    app = create_simple_app()
    
    # Add callback after server setup
    def add_callback():
        try:
            pn.state.add_periodic_callback(global_gui.update_progress_display, 2000)
            print("Progress monitoring callback added successfully")
        except Exception as e:
            print(f"Warning: Could not add progress callback: {e}")

    # Schedule callback to be added when server starts
    pn.state.onload(add_callback)

    # Add cleanup callback for when server shuts down
    def cleanup_on_exit():
        try:
            if AI_ASSISTANT_AVAILABLE:
                ai_config.cleanup_session()
                print("AI session cleanup completed")
        except Exception as e:
            print(f"Warning: AI cleanup failed: {e}")

    # Register cleanup for server shutdown
    import atexit
    atexit.register(cleanup_on_exit)

    # Serve the app for JupyterHub
    pn.serve(
        app,
        port=5000,  # Use port 5000 as requested
        allow_websocket_origin=["*"],  # Allow JupyterHub proxy
        show=False,  # Don't auto-open browser
        title="TensorSwitch GUI v2.0 - Multi-Page Interface"
    )