"""
Project Manager - Organize scanned objects into categories and projects
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path

class ProjectManager:
    """Manages scan projects with organized folder structure."""
    
    def __init__(self, base_dir=None):
        """Initialize project manager.
        
        Args:
            base_dir: Base directory for camera tools. If None, uses script location.
        """
        if base_dir is None:
            # Auto-detect camera_tools directory
            base_dir = Path(__file__).parent.parent
        
        self.base_dir = Path(base_dir)
        self.projects_dir = self.base_dir / "projects"
        self.projects_dir.mkdir(exist_ok=True)
        
        # Create default categories
        default_categories = ["furniture", "vehicles", "electronics", "misc", "new"]
        for category in default_categories:
            (self.projects_dir / category).mkdir(exist_ok=True)
    
    def create_project(self, name, category="misc", notes="", allow_overwrite=True):
        """Create a new scan project with organized folder structure.
        
        Args:
            name: Project name (will be sanitized)
            category: Category folder (furniture, vehicles, etc.)
            notes: Optional description/notes
            allow_overwrite: If True, prompt user for overwrite on duplicate names
            
        Returns:
            Path to created project directory, or None if cancelled
        """
        # Sanitize project name
        safe_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in name)
        safe_name = safe_name.strip().replace(' ', '_').lower()
        
        if not safe_name:
            safe_name = "unnamed_project"
        
        # Create category folder if needed
        category_dir = self.projects_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Look for existing projects with same base name
        existing_projects = []
        for proj_dir in category_dir.iterdir():
            if proj_dir.is_dir() and proj_dir.name.startswith(safe_name + "_"):
                existing_projects.append(proj_dir)
        
        project_dir = None
        
        # If projects exist and overwrite is allowed, prompt user
        if existing_projects and allow_overwrite:
            print(f"\n⚠️  Found {len(existing_projects)} existing project(s) with name '{name}':")
            for i, proj in enumerate(existing_projects, 1):
                metadata_file = proj / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            meta = json.load(f)
                            point_count = meta.get('point_count', 0)
                            created = meta.get('created', 'unknown')[:10]
                            print(f"  {i}. {proj.name}")
                            print(f"      {point_count} points | Created: {created}")
                    except:
                        print(f"  {i}. {proj.name} (metadata error)")
                else:
                    print(f"  {i}. {proj.name}")
            
            print("\n📋 Options:")
            print("  1. Overwrite existing project (⚠️  WARNING: deletes old data!)")
            print("  2. Create new timestamped project (RECOMMENDED - keeps both)")
            print("  3. Cancel")
            
            choice = input("\n👉 Choice (1-3) [default=2]: ").strip() or "2"
            
            if choice == '1':
                # User wants to overwrite - ask which one
                if len(existing_projects) == 1:
                    project_dir = existing_projects[0]
                    confirm = input(f"\n🗑️  Really DELETE all data in '{project_dir.name}'? (yes/no): ").strip().lower()
                    if confirm != 'yes':
                        print("   Cancelled - creating new project instead")
                        choice = '2'  # Fall through to create new
                else:
                    idx_choice = input(f"\n👉 Which project to overwrite (1-{len(existing_projects)}): ").strip()
                    if idx_choice.isdigit():
                        idx = int(idx_choice) - 1
                        if 0 <= idx < len(existing_projects):
                            project_dir = existing_projects[idx]
                            confirm = input(f"\n🗑️  Really DELETE all data in '{project_dir.name}'? (yes/no): ").strip().lower()
                            if confirm != 'yes':
                                print("   Cancelled - creating new project instead")
                                choice = '2'
                        else:
                            print("   Invalid choice - creating new project")
                            choice = '2'
                    else:
                        print("   Invalid input - creating new project")
                        choice = '2'
                
                if choice == '1' and project_dir:
                    # Clear old data
                    print(f"\n🗑️  Deleting old data from: {project_dir.name}")
                    for subfolder in ["raw_captures", "point_clouds", "previews", "reports"]:
                        folder_path = project_dir / subfolder
                        if folder_path.exists():
                            shutil.rmtree(folder_path)
                        folder_path.mkdir()
                    
                    print(f"✓ Overwriting project: {project_dir}")
                    
            elif choice == '3':
                print("❌ Cancelled")
                return None
        
        # Create new timestamped project (choice == '2' or no existing projects)
        if project_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = f"{safe_name}_{timestamp}"
            project_dir = category_dir / project_name
            project_dir.mkdir()
            
            (project_dir / "raw_captures").mkdir()
            (project_dir / "point_clouds").mkdir()
            (project_dir / "previews").mkdir()
            (project_dir / "reports").mkdir()
            
            print(f"✓ Created new project: {project_dir.name}")
        
        # Create/update metadata
        metadata = {
            "name": name,
            "category": category,
            "created": datetime.now().isoformat(),
            "notes": notes,
            "scan_settings": {},
            "point_count": 0,
            "frame_count": 0,
            "quality_score": 0.0
        }
        
        metadata_file = project_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📁 Project ready: {project_dir}")
        return str(project_dir)
    
    def update_project_metadata(self, project_dir, **kwargs):
        """Update project metadata with new information.
        
        Args:
            project_dir: Path to project directory
            **kwargs: Key-value pairs to update in metadata
        """
        project_path = Path(project_dir)
        metadata_file = project_path / "metadata.json"
        
        if not metadata_file.exists():
            print(f"⚠️  Metadata file not found: {metadata_file}")
            return
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Update fields
        metadata.update(kwargs)
        metadata["last_modified"] = datetime.now().isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def list_projects(self, category=None):
        """List all projects, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of project dictionaries with metadata
        """
        projects = []
        
        if category:
            search_dirs = [self.projects_dir / category]
        else:
            search_dirs = [d for d in self.projects_dir.iterdir() if d.is_dir()]
        
        for cat_dir in search_dirs:
            if not cat_dir.is_dir():
                continue
                
            for proj_dir in cat_dir.iterdir():
                if not proj_dir.is_dir():
                    continue
                    
                metadata_file = proj_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                            projects.append({
                                "path": str(proj_dir),
                                "name": metadata.get("name", proj_dir.name),
                                "category": metadata.get("category", "unknown"),
                                "created": metadata.get("created", "unknown"),
                                "point_count": metadata.get("point_count", 0),
                                "quality_score": metadata.get("quality_score", 0.0)
                            })
                    except:
                        pass  # Skip projects with corrupt metadata
        
        return sorted(projects, key=lambda x: x["created"], reverse=True)
    
    def get_categories(self):
        """Get list of all category folders.
        
        Returns:
            List of category names
        """
        categories = []
        for d in self.projects_dir.iterdir():
            if d.is_dir() and not d.name.startswith('.'):
                categories.append(d.name)
        return sorted(categories)
    
    def get_project_stats(self):
        """Get statistics about all projects.
        
        Returns:
            Dictionary with project statistics
        """
        projects = self.list_projects()
        
        total_points = sum(p["point_count"] for p in projects)
        avg_quality = sum(p["quality_score"] for p in projects) / len(projects) if projects else 0
        
        categories = {}
        for p in projects:
            cat = p["category"]
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_projects": len(projects),
            "total_points": total_points,
            "average_quality": avg_quality,
            "categories": categories
        }