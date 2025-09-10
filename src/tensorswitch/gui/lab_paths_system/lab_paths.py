#!/usr/bin/env python3
"""
Hierarchical Lab Paths Configuration for TensorSwitch GUI

This module parses the HHMI Lab and Project File Share Paths Excel file
and provides structured lab path mappings for hierarchical dropdown navigation.

Data flow: Lab → Storage Type → Platform → Path
Retrieved: 2025-09-09
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class HierarchicalLabPaths:
    """Manager for hierarchical lab paths from HHMI Excel file"""
    
    def __init__(self, excel_file: str = None):
        if excel_file is None:
            # Default to Excel file in the same directory as this script
            current_dir = Path(__file__).parent
            excel_file = str(current_dir / "Lab_and_project_file_share_path.xlsx")
        self.excel_file = excel_file
        self.data_retrieved = "2025-09-09"  # Today's date
        self.lab_data = {}
        self.all_lab_names = []
        self._load_and_parse_data()
    
    def _load_and_parse_data(self):
        """Load and parse the Excel file into hierarchical structure"""
        try:
            df = pd.read_excel(self.excel_file)
            
            # Find the header row (row 8 contains: Lab | Storage | Mac | Windows | Cluster | AD groups)
            header_row_idx = None
            for i, row in df.iterrows():
                if (pd.notna(row.iloc[0]) and str(row.iloc[0]).strip().lower() == 'lab' and
                    pd.notna(row.iloc[1]) and 'storage' in str(row.iloc[1]).lower()):
                    header_row_idx = i
                    break
            
            if header_row_idx is None:
                raise ValueError("Could not find header row with 'Lab | Storage | Mac | Windows | Cluster | AD groups'")
            
            # Parse data starting after header row
            current_lab = None
            
            for i, row in df.iterrows():
                if i <= header_row_idx:  # Skip header and above
                    continue
                
                lab_name = row.iloc[0] if pd.notna(row.iloc[0]) else None
                storage_type = row.iloc[1] if pd.notna(row.iloc[1]) else None
                mac_path = row.iloc[2] if pd.notna(row.iloc[2]) else ""
                windows_path = row.iloc[3] if pd.notna(row.iloc[3]) else ""
                cluster_path = row.iloc[4] if pd.notna(row.iloc[4]) else ""
                ad_group = row.iloc[5] if pd.notna(row.iloc[5]) else ""
                
                # Skip empty rows
                if not any([lab_name, storage_type, cluster_path]):
                    continue
                
                # Update current lab
                if lab_name:
                    current_lab = str(lab_name).strip()
                
                if current_lab and storage_type and cluster_path:
                    # Initialize lab if not exists
                    if current_lab not in self.lab_data:
                        self.lab_data[current_lab] = {
                            'storage_types': {},
                            'ad_group': ad_group,
                            'description': f"{current_lab} Lab"
                        }
                    
                    # Add storage type data
                    storage_type = str(storage_type).strip()
                    self.lab_data[current_lab]['storage_types'][storage_type] = {
                        'platforms': {
                            'mac': str(mac_path).strip() if mac_path else "",
                            'windows': str(windows_path).strip() if windows_path else "",
                            'cluster': str(cluster_path).strip() if cluster_path else ""
                        },
                        'ad_group': ad_group
                    }
            
            # Extract all lab names
            self.all_lab_names = sorted(self.lab_data.keys())
            print(f"Successfully loaded {len(self.all_lab_names)} labs from Excel file")
            
        except Exception as e:
            print(f"Warning: Could not load Excel file {self.excel_file}: {e}")
            self._create_fallback_data()
    
    def _create_fallback_data(self):
        """Create fallback data if Excel file can't be read"""
        fallback_labs = [
            "ahrens", "branson", "murphy", "mengwang", 
            "keller", "tavakoli", "scicompsoft"
        ]
        
        for lab in fallback_labs:
            if lab == "scicompsoft":
                cluster_path = f"/groups/{lab}"
            else:
                cluster_path = f"/nrs/{lab}"
                
            self.lab_data[lab] = {
                'storage_types': {
                    'primary': {
                        'platforms': {
                            'cluster': cluster_path,
                            'mac': f"/Volumes/{lab}",
                            'windows': f"\\\\prfs.hhmi.org\\{lab}"
                        },
                        'ad_group': lab
                    }
                },
                'ad_group': lab,
                'description': f"{lab.title()} Lab"
            }
        
        self.all_lab_names = sorted(fallback_labs)
    
    def get_lab_names(self) -> List[str]:
        """Get sorted list of all lab names"""
        return self.all_lab_names.copy()
    
    def get_storage_types(self, lab_name: str) -> List[str]:
        """Get available storage types for a specific lab"""
        if lab_name not in self.lab_data:
            return []
        return list(self.lab_data[lab_name]['storage_types'].keys())
    
    def get_platforms(self) -> List[str]:
        """Get available platforms (mac, windows, cluster)"""
        return ['cluster', 'mac', 'windows']
    
    def get_path(self, lab_name: str, storage_type: str, platform: str) -> str:
        """Get specific path for lab/storage/platform combination"""
        if (lab_name not in self.lab_data or 
            storage_type not in self.lab_data[lab_name]['storage_types']):
            return ""
        
        platforms = self.lab_data[lab_name]['storage_types'][storage_type]['platforms']
        return platforms.get(platform, "")
    
    def get_ad_group(self, lab_name: str) -> str:
        """Get AD group (project name) for a lab"""
        if lab_name not in self.lab_data:
            return ""
        return self.lab_data[lab_name].get('ad_group', lab_name.lower())
    
    def get_all_project_names(self) -> List[str]:
        """Get all unique AD group names for project billing"""
        projects = set()
        for lab_name in self.lab_data:
            ad_group = self.get_ad_group(lab_name)
            if ad_group:
                projects.add(ad_group)
        
        # Add empty string for default
        project_list = [""] + sorted(list(projects))
        return project_list
    
    def suggest_common_subdirs(self, base_path: str) -> List[str]:
        """Suggest common subdirectories for a base path"""
        common_subdirs = [
            "data", "results", "raw_data", "processed_data", 
            "datasets", "imaging", "microscopy", "analysis", 
            "projects", "shared", "work"
        ]
        
        if not base_path:
            return []
            
        return [f"{base_path}/{subdir}" for subdir in common_subdirs]
    
    def get_lab_summary(self, lab_name: str) -> Dict:
        """Get complete summary for a lab"""
        if lab_name not in self.lab_data:
            return {}
        
        lab_info = self.lab_data[lab_name].copy()
        lab_info['name'] = lab_name
        lab_info['project'] = self.get_ad_group(lab_name)
        return lab_info
    
    def search_labs(self, query: str) -> List[str]:
        """Search labs by name (case-insensitive)"""
        query = query.lower().strip()
        if not query:
            return self.get_lab_names()
        
        matches = []
        for lab_name in self.all_lab_names:
            if query in lab_name.lower():
                matches.append(lab_name)
        
        return matches
    
    def export_data(self, output_file: str = 'hierarchical_lab_paths.json'):
        """Export parsed data to JSON for debugging"""
        export_data = {
            'data_retrieved': self.data_retrieved,
            'total_labs': len(self.all_lab_names),
            'lab_data': self.lab_data,
            'all_lab_names': self.all_lab_names
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"Hierarchical lab paths exported to {output_file}")
    
    def get_platform_display_names(self) -> Dict[str, str]:
        """Get user-friendly platform names"""
        return {
            'cluster': 'Cluster/Linux',
            'mac': 'Mac',
            'windows': 'Windows/Linux SMB'
        }
    
    def validate_path_exists(self, path: str) -> bool:
        """Basic validation if a cluster path might exist"""
        if not path or not path.startswith('/'):
            return False
        
        # Basic validation - cluster paths should start with /groups or /nrs
        return path.startswith('/groups/') or path.startswith('/nrs/')

# Global instance
_hierarchical_lab_paths = None

def get_lab_paths() -> HierarchicalLabPaths:
    """Get global hierarchical lab paths instance"""
    global _hierarchical_lab_paths
    if _hierarchical_lab_paths is None:
        _hierarchical_lab_paths = HierarchicalLabPaths()
    return _hierarchical_lab_paths

# Convenience functions for GUI integration
def get_all_labs() -> List[str]:
    """Get all lab names for dropdown"""
    return get_lab_paths().get_lab_names()

def get_lab_storage_types(lab_name: str) -> List[str]:
    """Get storage types for a lab"""
    return get_lab_paths().get_storage_types(lab_name)

def get_available_platforms() -> List[str]:
    """Get available platforms"""
    return get_lab_paths().get_platforms()

def get_lab_path(lab_name: str, storage_type: str, platform: str) -> str:
    """Get path for lab/storage/platform combination"""
    return get_lab_paths().get_path(lab_name, storage_type, platform)

def get_lab_project(lab_name: str) -> str:
    """Get project name for billing"""
    return get_lab_paths().get_ad_group(lab_name)

def get_all_projects() -> List[str]:
    """Get all project names for billing dropdown"""
    return get_lab_paths().get_all_project_names()

def suggest_subdirs(base_path: str) -> List[str]:
    """Get suggested subdirectories"""
    return get_lab_paths().suggest_common_subdirs(base_path)

if __name__ == "__main__":
    # Test the hierarchical parser
    manager = HierarchicalLabPaths()
    manager.export_data()
    
    print(f"Data retrieved: {manager.data_retrieved}")
    print(f"Total labs: {len(manager.get_lab_names())}")
    print("\nFirst 10 labs:", manager.get_lab_names()[:10])
    
    # Test hierarchical access
    if manager.get_lab_names():
        test_lab = manager.get_lab_names()[0]
        print(f"\nTesting lab: {test_lab}")
        print(f"Storage types: {manager.get_storage_types(test_lab)}")
        
        if manager.get_storage_types(test_lab):
            test_storage = manager.get_storage_types(test_lab)[0]
            print(f"Testing storage type: {test_storage}")
            for platform in manager.get_platforms():
                path = manager.get_path(test_lab, test_storage, platform)
                print(f"  {platform}: {path}")
        
        print(f"Project/AD group: {manager.get_ad_group(test_lab)}")
    
    print(f"\nTotal unique projects: {len(manager.get_all_project_names())}")
    print("First 10 projects:", manager.get_all_project_names()[:10])