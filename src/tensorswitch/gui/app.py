#!/usr/bin/env python3
"""
Simple test version of TensorSwitch GUI for testing basic functionality
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

class SimpleTensorSwitchGUI(param.Parameterized):
    """Multi-page GUI for TensorSwitch with welcome page and structured workflow"""
    
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

        # Initialize format detection system
        self._setup_format_detection()

        # Initialize conversion plan
        self.conversion_plan = None

        # Initialize lab paths system and populate project dropdown
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
        
        # Start with welcome page
        self.show_welcome_page()
        
    def create_welcome_page(self):
        """Create the welcome/landing page"""
        # Welcome title with switch emoji
        welcome_title = pn.pane.Markdown("""
        <div style="text-align: center; margin: 60px 0;">
        <h1 style="font-size: 3.5em; margin-bottom: 20px; color: #2c3e50;">
        Welcome to TensorSwitch! 🔄
        </h1>
        <p style="font-size: 1.3em; color: #7f8c8d; margin-bottom: 40px;">
        <strong>Convert scientific data formats effortlessly at Janelia Research Campus</strong>
        </p>
        </div>
        """)
        
        # Explanation section
        explanation = pn.pane.Markdown("""
        <div style="background: #f8f9fa; padding: 40px; border-radius: 15px; margin: 40px 0; text-align: center;">
        <h2 style="color: #34495e; margin-bottom: 25px;">What is TensorSwitch?</h2>
        <p style="font-size: 1.1em; line-height: 1.6; color: #2c3e50; margin-bottom: 20px;">
        TensorSwitch is a high-performance scientific data conversion tool designed for large-scale microscopy and imaging datasets. 
        It efficiently converts between formats like <strong>TIFF, ND2, IMS, N5, and Zarr</strong> while preserving metadata and enabling parallel processing.
        </p>
        <p style="font-size: 1.1em; line-height: 1.6; color: #2c3e50; margin-bottom: 20px;">
        This GUI makes TensorSwitch accessible to all researchers - no command-line experience required! 
        Simply specify your input data, choose the desired output format, and let TensorSwitch handle the conversion.
        </p>
        <div style="background: #e8f5e8; padding: 20px; border-radius: 8px; margin-top: 30px;">
        <h3 style="color: #27ae60; margin-bottom: 15px;">✨ Key Features:</h3>
        <p style="color: #2c3e50;">
        • Support for 10 conversion tasks • Zarr2 & Zarr3 format support<br>
        • Local and cluster execution • Dask JobQueue integration<br>
        • Real-time progress tracking • Automatic OME-Zarr multiscale metadata<br>
        • Chunked processing for large datasets • LSF job submission
        </p>
        </div>
        </div>
        """)
        
        # Let's convert button
        self.start_btn = pn.widgets.Button(
            name="🚀 Let's Convert!", 
            button_type="primary",
            width=250,
            height=60,
            styles={'font-size': '1.3em', 'margin': '20px auto', 'display': 'block'}
        )
        self.start_btn.on_click(self.show_conversion_page)
        
        self.welcome_layout = pn.Column(
            welcome_title,
            explanation,
            pn.Row(pn.Spacer(), self.start_btn, pn.Spacer()),
            sizing_mode="stretch_width",
            styles={'max-width': '900px', 'margin': '0 auto', 'padding': '40px 20px'}
        )
        
    def create_conversion_page(self):
        """Create the main conversion configuration page with clear step-by-step workflow"""
        # Page header with progress steps
        header = pn.pane.Markdown("""
        # 🔬 TensorSwitch Conversion Workflow

        <div style="background: #e3f2fd; padding: 20px; border-radius: 10px; margin-bottom: 30px;">
        <h3 style="margin-top: 0; color: #1976d2;">📋 Simple 4-Step Process:</h3>
        <p style="margin-bottom: 0; font-size: 1.1em;">
        <strong>Step 1:</strong> Enter file paths → <strong>Step 2:</strong> Choose workflow mode → <strong>Step 3:</strong> Configure conversion → <strong>Step 4:</strong> Execute
        </p>
        </div>
        """, styles={'text-align': 'center', 'margin-bottom': '20px'})
        
        # STEP 1: File paths section with clear numbering
        step1_header = pn.pane.Markdown("""
        ## 📁 Step 1: Specify Input and Output Paths
        """, styles={'color': '#1976d2', 'margin-bottom': '15px'})

        file_hints = pn.pane.Markdown("""
        **💡 Path Hints:**
        - **PRFS storage**: `/groups/[lab_name]/...`
        - **NRS storage**: `/nrs/[lab_name]/...`
        - **Wiki Lab Path**: [Lab and Project File Share Paths](https://hhmi.atlassian.net/wiki/spaces/SCS/pages/152469629/Lab+and+Project+File+Share+Paths)
        """, styles={'background': '#fff3cd', 'padding': '15px', 'border-radius': '8px', 'margin-bottom': '15px'})
        
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
                print(f"Could not create path selector: {e}")
                path_selector_widget = pn.pane.Markdown("*Path helper not available*")
        else:
            path_selector_widget = pn.pane.Markdown("*Path helper not available*")
        
        file_section = pn.Column(
            step1_header,
            file_hints,
            path_selector_widget,
            pn.layout.Divider(),
            self.input_widget,
            self.output_widget,
            styles={'background': '#ffffff', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px'}
        )
        
        # STEP 2: Workflow mode selection
        step2_header = pn.pane.Markdown("""
        ## 🔄 Step 2: Choose Your Workflow Mode
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
            styles={'background': '#ffffff', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px'}
        )

        # STEP 3A: Format detection section (dynamic - only for smart mode)
        format_header = pn.pane.Markdown("""
        ## 🔍 Step 3A: Smart Format Detection (Smart Mode Only)
        """, styles={'color': '#1976d2', 'margin-bottom': '15px'})
        format_content = pn.pane.Markdown(
            "**🔄 Format Detection**: Enter input path above to auto-detect format and view file metadata",
            styles={'background': '#e3f2fd', 'padding': '15px', 'border-radius': '8px', 'margin-bottom': '20px'}
        )
        self.format_section = pn.Column(format_header, format_content)

        # STEP 3B: Output format selection section (smart mode only)
        output_header = pn.pane.Markdown("""
        ## 📤 Step 3B: Configure Output Format (Smart Mode Only)
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
            styles={'background': '#ffffff', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px'}
        )

        # Conversion plan section (dynamic)
        plan_header = pn.pane.Markdown("### 🗂️ Conversion Plan")
        plan_content = pn.pane.Markdown(
            "**Conversion Plan**: Configure input and output to see execution plan",
            styles={'background': '#f8f9fa', 'padding': '15px', 'border-radius': '8px', 'margin-bottom': '20px'}
        )
        self.plan_section = pn.Column(plan_header, plan_content)
        
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
        
        # STEP 3C: Task selection section (manual mode only)
        manual_task_header = pn.pane.Markdown("""
        ## ⚙️ Step 3: Manual Task Selection (Manual Mode Only)
        """, styles={'color': '#1976d2', 'margin-bottom': '15px'})

        self.task_section = pn.Column(
            manual_task_header,
            task_description,
            pn.Param(
                self,
                parameters=['task'],
                widgets={'task': {'type': pn.widgets.Select, 'name': 'Select conversion task'}}
            ),
            styles={'background': '#ffffff', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px'},
            visible=False  # Start hidden since we default to smart mode
        )
        
        # Execution location with clear options
        execution_description = pn.pane.Markdown("""
        **Choose where to run your conversion:**
        - **Run locally**: Process on current machine (good for small files, immediate results)
        - **Submit to cluster**: Use LSF cluster (recommended for large files, requires project billing)
        """, styles={'background': '#f8f9fa', 'padding': '10px', 'border-radius': '5px', 'margin-bottom': '10px'})
        
        # STEP 4: Execution configuration
        step4_header = pn.pane.Markdown("""
        ## 🚀 Step 4: Configure Execution & Run
        """, styles={'color': '#1976d2', 'margin-bottom': '15px'})

        execution_section = pn.Column(
            step4_header,
            execution_description,
            pn.Param(
                self,
                parameters=['run_locally'],
                widgets={'run_locally': {'type': pn.widgets.Checkbox, 'name': 'Run locally (uncheck to submit to cluster)'}}
            ),
            styles={'background': '#f8f9fa', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px'}
        )
        
        # Conditional parameters containers
        self.local_params = pn.Column(
            pn.pane.Markdown("### 🔧 Local Processing Options"),
            pn.Param(
                self,
                parameters=['use_ome_structure', 'use_shard', 'level', 'memory_limit'],
                widgets={
                    'use_ome_structure': {'type': pn.widgets.Checkbox, 'name': 'Use OME-Zarr structure'},
                    'use_shard': {'type': pn.widgets.Checkbox, 'name': 'Use sharded format (Zarr3 only)'},
                    'level': {'type': pn.widgets.IntSlider, 'name': 'Downsampling level'},
                    'memory_limit': {'type': pn.widgets.IntSlider, 'name': 'Memory limit (%)'},
                }
            ),
            pn.pane.Markdown("**🎛️ Advanced Shape Parameters (Optional):**"),
            pn.Param(
                self,
                parameters=['custom_shard_shape', 'custom_chunk_shape'],
                widgets={
                    'custom_shard_shape': {'type': pn.widgets.TextInput, 'name': 'Custom shard shape (e.g., 128,576,576)', 'placeholder': 'Leave empty for defaults'},
                    'custom_chunk_shape': {'type': pn.widgets.TextInput, 'name': 'Custom chunk shape (e.g., 32,32,32)', 'placeholder': 'Leave empty for defaults'},
                }
            ),
            styles={'background': '#e8f5e8', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px'}
        )
        
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
            styles={'background': '#fff3cd', 'padding': '20px', 'border-radius': '10px', 'margin-bottom': '20px'},
            visible=False
        )
        
        # Action buttons
        self.back_btn = pn.widgets.Button(name="← Back to Welcome", button_type="light")
        self.preview_btn = pn.widgets.Button(name="🔍 Preview Job", button_type="light")
        self.submit_btn = pn.widgets.Button(name="🚀 Run Job", button_type="primary")
        
        self.back_btn.on_click(self.show_welcome_page)
        self.preview_btn.on_click(self.preview_job)
        self.submit_btn.on_click(self.submit_job)
        
        button_row = pn.Row(
            self.back_btn,
            pn.Spacer(),
            self.preview_btn, 
            self.submit_btn,
            styles={'margin': '20px 0'}
        )
        
        # Preview box
        self.preview_box = pn.pane.Markdown(
            "**Job Preview**: Click '🔍 Preview Job' to see job details",
            styles={'background': '#e3f2fd', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #2196f3', 'margin-bottom': '15px'}
        )
        
        self.conversion_layout = pn.Column(
            header,
            file_section,
            mode_section,
            self.format_section,
            self.output_section,
            self.plan_section,
            self.task_section,
            execution_section,
            self.local_params,
            self.cluster_params,
            button_row,
            self.preview_box,
            sizing_mode="stretch_width",
            styles={'max-width': '800px', 'margin': '0 auto', 'padding': '20px'}
        )

        # Set initial visibility based on workflow mode
        self._update_workflow_visibility()
        
        # Watch for run_locally changes to show/hide cluster options
        self.param.watch(self.update_execution_options, 'run_locally')
        # Watch for task changes to handle Zarr2-specific logic
        self.param.watch(self.update_task_options, 'task')
        # Watch for input path changes to trigger format detection
        self.input_widget.param.watch(self.analyze_input_file, 'value')
        # Watch for output configuration changes to update conversion plan
        self.param.watch(self.update_conversion_plan, 'output_format')
        self.param.watch(self.update_conversion_plan, 'max_downsample_level')
        # Watch for workflow mode changes
        self.param.watch(self._update_workflow_visibility, 'workflow_mode')
        
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
        """Show the conversion configuration page"""
        self.current_page = "conversion"
        # Clear main container and add conversion content
        self.main_container.clear()
        self.main_container.append(self.conversion_layout)
        
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
            content = f"**Auto-Detected Input**:\n{summary}"
            style = {'background': '#d4edda', 'padding': '15px', 'border-radius': '8px'}
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
        """Update the conversion plan section with generated plan"""
        if not self.conversion_plan or not hasattr(self, 'plan_section'):
            return

        if self.conversion_plan.get('error'):
            # Show error
            content = f"**Plan Error**: ❌ {self.conversion_plan['error']}"
            style = {'background': '#f8d7da', 'padding': '15px', 'border-radius': '8px'}
        elif self.conversion_plan.get('tasks'):
            # Show conversion plan
            file_size_mb = self.input_analysis.get('size_mb', 0)
            plan_summary = self.task_planner.format_plan_summary(self.conversion_plan, file_size_mb)
            content = f"**Execution Plan**:\n{plan_summary}"
            style = {'background': '#d4edda', 'padding': '15px', 'border-radius': '8px'}
        else:
            content = "**Conversion Plan**: Configure input and output to see execution plan"
            style = {'background': '#f8f9fa', 'padding': '15px', 'border-radius': '8px'}

        # Update plan section content
        self.plan_section[1].object = content
        self.plan_section[1].styles = style

    def _update_workflow_visibility(self, *args, **kwargs):
        """Update section visibility based on workflow mode"""
        is_smart_mode = self.workflow_mode == "smart"

        # Smart mode sections
        self.format_section.visible = is_smart_mode
        self.output_section.visible = is_smart_mode
        self.plan_section.visible = is_smart_mode

        # Manual mode sections
        self.task_section.visible = not is_smart_mode

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
            
        if not self.project:
            self.preview_box.object = "**Job Preview**: ⚠️ Please select a project for billing"
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

        preview_text = f"""
**Job Preview**: 🔍

- **Task**: `{self.task}`{format_note}
- **Input**: `{input_path}`
- **Output**: `{output_path}`
- **Project**: `{self.project}`
- **Execution**: Run {execution_mode}{dask_note}
- **Level**: {self.level}
- **Use Sharding**: {self.use_shard}{sharding_note}
- **OME Structure**: {self.use_ome_structure} {'(auto multiscale metadata)' if self.use_ome_structure else ''}
- **CPU Cores**: {self.cores}
- **Wall Time**: {self.wall_time}
- **Parallel Volumes**: {self.num_volumes}
- **Memory Limit**: {self.memory_limit}%

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
        """Execute local job with real-time progress monitoring"""
        try:
            # Build command for local execution
            cmd = [
                "python", "-m", "tensorswitch",
                "--task", self.task,
                "--base_path", input_path,
                "--output_path", output_path,
                "--level", str(self.level),
                "--use_shard", "1" if self.use_shard else "0",
                "--use_ome_structure", "1" if self.use_ome_structure else "0",
                "--memory_limit", str(self.memory_limit),
                "--num_volumes", str(self.num_volumes)
            ]
            
            # Show command being executed
            cmd_str = " ".join(cmd)
            self.status.object = f"**Status**: 🟡 Starting local job...\n\n`{cmd_str}`\n\n📊 Analyzing dataset..."
            self.status.styles = {'background': '#fff3cd', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #ffc107'}
            
            # Execute command with real-time output capture
            self.current_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True, 
                bufsize=1,
                universal_newlines=True,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
            
            # Monitor progress in real-time
            self._monitor_job_progress()
            
        except Exception as e:
            self._job_finished(success=False, message=f"Error starting job: {str(e)}")
    
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
    
    def submit_cluster_job(self, input_path, output_path):
        """Submit job to LSF cluster and show job tracking info in preview box"""
        # Update preview box to show submission is starting
        self.preview_box.object = f"**Job Submission**: 🚀 Submitting to LSF cluster...\n\nInput: `{input_path}`\nOutput: `{output_path}`\nProject: `{self.project}`"
        self.preview_box.styles = {'background': '#fff3cd', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #ffc107', 'margin-bottom': '15px'}
        
        try:
            # Build command for cluster submission based on CLI implementation
            cmd = [
                "python", "-m", "tensorswitch",
                "--task", self.task,
                "--base_path", input_path,
                "--output_path", output_path,
                "--level", str(self.level),
                "--use_shard", "1" if self.use_shard else "0",
                "--use_ome_structure", "1" if self.use_ome_structure else "0",
                "--memory_limit", str(self.memory_limit),
                "--cores", str(self.cores),
                "--wall_time", str(self.wall_time),
                "--num_volumes", str(self.num_volumes),
                "--project", self.project,
                "--submit"  # This triggers cluster submission
            ]

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
                
                # Create job tracking information
                job_info = ""
                if job_ids:
                    job_info = f"\n**📋 Job Tracking:**\n"
                    for i, (job_id, job_name) in enumerate(zip(job_ids, job_names)):
                        job_info += f"- **Job {i+1}**: `{job_name}` (ID: {job_id})\n"
                    job_info += f"\n**🔍 Check Status:** `bjobs {' '.join(job_ids)}`\n"
                    job_info += f"**📊 Monitor:** `bjobs -w` or `bjobs -l {job_ids[0]}`\n"
                    job_info += f"**📁 Logs:** `/groups/scicompsoft/home/chend/temp/downsample_script/tensorswitch/output/`\n"
                
                self.preview_box.object = f"""**✅ Cluster Jobs Submitted Successfully!**

**Input:** `{input_path}`  
**Output:** `{output_path}`  
**Project:** `{self.project}`  
**Resources:** {self.cores} cores, {self.wall_time} wall time{job_info}

**💡 Tip:** Jobs are running in the background. Check output directory when complete!"""
                
                self.preview_box.styles = {'background': '#d4edda', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #28a745', 'margin-bottom': '15px'}
            else:
                self.preview_box.object = f"**❌ Cluster Submission Failed!**\n\n**Error Details:**\n```\n{result.stderr[-800:] if result.stderr else result.stdout[-800:] if result.stdout else 'No error details available'}\n```"
                self.preview_box.styles = {'background': '#f8d7da', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #dc3545', 'margin-bottom': '15px'}
                
        except Exception as e:
            self.preview_box.object = f"**❌ Error Submitting Cluster Job**\n\n{str(e)}"
            self.preview_box.styles = {'background': '#f8d7da', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #dc3545', 'margin-bottom': '15px'}

def create_simple_app():
    """Create simple test app"""
    pn.extension()
    
    gui = SimpleTensorSwitchGUI()
    
    # Store gui reference globally so callback can be added later
    global global_gui
    global_gui = gui
    
    return gui.layout

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
    
    # Serve the app for JupyterHub
    pn.serve(
        app,
        port=5008,  # Use different port
        allow_websocket_origin=["*"],  # Allow JupyterHub proxy
        show=False,  # Don't auto-open browser
        title="TensorSwitch GUI v2.0 - Multi-Page Interface"
    )