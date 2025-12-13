"""
Object Image Management System for 3D Scanning
Organizes captured images by object name with easy recall
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

class ObjectManager:
    """Manage scanned object folders and metadata."""
    
    def __init__(self, base_dir="D:/Users/Planet UI/3d_scan_objects"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.catalog_file = self.base_dir / "object_catalog.json"
        self.catalog = self.load_catalog()
    
    def load_catalog(self):
        """Load object catalog from JSON."""
        if self.catalog_file.exists():
            with open(self.catalog_file, 'r') as f:
                return json.load(f)
        return {"objects": [], "sessions": {}}
    
    def save_catalog(self):
        """Save object catalog to JSON."""
        with open(self.catalog_file, 'w') as f:
            json.dump(self.catalog, f, indent=2)
    
    def create_object_folder(self, object_name, category="uncategorized"):
        """
        Create a new object folder structure.
        
        Args:
            object_name: Name for the object (e.g., "ford_mustang_1978")
            category: Category folder (e.g., "vehicles", "furniture")
        
        Returns:
            Path to created object folder
        """
        # Sanitize name
        safe_name = "".join(c for c in object_name if c.isalnum() or c in (' ', '_', '-')).strip()
        safe_name = safe_name.replace(' ', '_').lower()
        
        # Create folder structure
        category_path = self.base_dir / category
        object_path = category_path / safe_name
        
        folders = {
            'raw_images': object_path / 'raw_images',
            'processed': object_path / 'processed',
            'point_clouds': object_path / 'point_clouds',
            'meshes': object_path / 'meshes',
            'calibration': object_path / 'calibration'
        }
        
        for folder in folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file
        metadata = {
            'name': object_name,
            'safe_name': safe_name,
            'category': category,
            'created': datetime.now().isoformat(),
            'sessions': [],
            'total_images': 0,
            'last_modified': datetime.now().isoformat()
        }
        
        metadata_file = object_path / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Add to catalog
        if safe_name not in [obj['safe_name'] for obj in self.catalog['objects']]:
            self.catalog['objects'].append({
                'name': object_name,
                'safe_name': safe_name,
                'category': category,
                'path': str(object_path),
                'created': metadata['created']
            })
            self.save_catalog()
        
        print(f"✅ Created object folder: {object_path}")
        return object_path
    
    def start_capture_session(self, object_name, session_name=None):
        """
        Start a new capture session for an object.
        
        Args:
            object_name: Name of object (must exist)
            session_name: Optional session name (auto-generated if None)
        
        Returns:
            Session info dict
        """
        # Find object
        object_info = self.find_object(object_name)
        if not object_info:
            print(f"❌ Object '{object_name}' not found!")
            print("\nAvailable objects:")
            self.list_objects()
            return None
        
        object_path = Path(object_info['path'])
        
        # Verify the path exists
        if not object_path.exists():
            print(f"❌ Object path does not exist: {object_path}")
            return None
        
        # Generate session name
        if session_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_name = f"session_{timestamp}"
        
        # Create session folder
        session_path = object_path / 'raw_images' / session_name
        session_path.mkdir(parents=True, exist_ok=True)
        
        session_info = {
            'name': session_name,
            'path': str(session_path),
            'started': datetime.now().isoformat(),
            'image_count': 0,
            'object_name': object_info['name'],
            'object_safe_name': object_info['safe_name']
        }
        
        # Update object metadata
        metadata_file = object_path / 'metadata.json'
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                'name': object_info['name'],
                'safe_name': object_info['safe_name'],
                'category': object_info['category'],
                'created': object_info['created'],
                'sessions': [],
                'total_images': 0,
                'last_modified': datetime.now().isoformat()
            }
        
        metadata['sessions'].append(session_info)
        metadata['last_modified'] = datetime.now().isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update catalog
        self.catalog['sessions'][session_name] = session_info
        self.save_catalog()
        
        print(f"✅ Started session: {session_name}")
        print(f"   Object: {object_info['name']}")
        print(f"   Images will be saved to: {session_path}")
        
        return session_info
    
    def save_image(self, frame, session_info, image_name=None):
        """Save an image to the current session."""
        if session_info is None:
            print("❌ No active session!")
            return None
        
        session_path = Path(session_info['path'])
        
        # Auto-generate image name
        if image_name is None:
            image_count = len(list(session_path.glob('*.png')))
            image_name = f"image_{image_count:04d}.png"
        
        image_path = session_path / image_name
        cv2.imwrite(str(image_path), frame)
        
        # Update session info
        session_info['image_count'] += 1
        
        print(f"✅ Saved: {image_name}")
        return image_path
    
    def end_capture_session(self, session_info):
        """End the current capture session."""
        if session_info is None:
            return
        
        session_info['ended'] = datetime.now().isoformat()
        
        # Update metadata
        object_info = self.find_object(session_info['object_name'])
        if not object_info:
            print(f"⚠️  Warning: Could not find object {session_info['object_name']}")
            return
            
        object_path = Path(object_info['path'])
        metadata_file = object_path / 'metadata.json'
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        metadata['total_images'] += session_info['image_count']
        metadata['last_modified'] = datetime.now().isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Session ended: {session_info['name']}")
        print(f"   Total images captured: {session_info['image_count']}")
    
    def find_object(self, object_name):
        """Find an object by name or safe_name."""
        for obj in self.catalog['objects']:
            if obj['name'] == object_name or obj['safe_name'] == object_name:
                return obj
        return None
    
    def list_objects(self, category=None):
        """List all objects, optionally filtered by category."""
        objects = self.catalog['objects']
        
        if category:
            objects = [obj for obj in objects if obj['category'] == category]
        
        if not objects:
            print("No objects found.")
            return []
        
        print("\n" + "="*70)
        print("SCANNED OBJECTS")
        print("="*70)
        
        for i, obj in enumerate(objects, 1):
            print(f"\n{i}. {obj['name']}")
            print(f"   Category: {obj['category']}")
            print(f"   Safe name: {obj['safe_name']}")
            print(f"   Path: {obj['path']}")
            print(f"   Created: {obj['created']}")
        
        print("="*70)
        return objects
    
    def list_sessions(self, object_name):
        """List all capture sessions for an object."""
        object_info = self.find_object(object_name)
        if not object_info:
            print(f"❌ Object '{object_name}' not found!")
            return []
        
        object_path = Path(object_info['path'])
        metadata_file = object_path / 'metadata.json'
        
        if not metadata_file.exists():
            print(f"⚠️  No metadata found for '{object_name}'")
            return []
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        sessions = metadata.get('sessions', [])
        
        if not sessions:
            print(f"No sessions found for '{object_name}'")
            return []
        
        print("\n" + "="*70)
        print(f"CAPTURE SESSIONS - {object_name}")
        print("="*70)
        
        for i, session in enumerate(sessions, 1):
            print(f"\n{i}. {session['name']}")
            print(f"   Started: {session['started']}")
            print(f"   Images: {session['image_count']}")
            if 'ended' in session:
                print(f"   Ended: {session['ended']}")
        
        print("="*70)
        return sessions
    
    def verify_and_fix_catalog(self):
        """Verify catalog matches filesystem and fix any issues."""
        print("\n🔍 Verifying catalog...")
        
        fixed_objects = []
        issues_found = 0
        
        for obj in self.catalog['objects']:
            obj_path = Path(obj['path'])
            
            # Check if path exists
            if not obj_path.exists():
                print(f"⚠️  Path not found: {obj['path']}")
                issues_found += 1
                continue
            
            # Check if metadata exists
            metadata_file = obj_path / 'metadata.json'
            if not metadata_file.exists():
                print(f"⚠️  Missing metadata: {obj['name']}")
                issues_found += 1
                continue
            
            # Check if sessions are valid
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            for session in metadata.get('sessions', []):
                if 'path' not in session or not Path(session['path']).exists():
                    print(f"⚠️  Invalid session path: {session['path']}")
                    issues_found += 1
                    continue
            
            fixed_objects.append(obj)
        
        # Update catalog with fixed objects
        self.catalog['objects'] = fixed_objects
        self.save_catalog()
        
        if issues_found == 0:
            print("✅ No issues found.")
        else:
            print(f"✅ Fixed catalog. Issues found and resolved: {issues_found}")
    
    def export_object(self, object_name, export_path):
        """
        Export an object folder to a zip file.
        
        Args:
            object_name: Name of the object to export
            export_path: Destination zip file path
        
        Returns:
            Path to the created zip file
        """
        # Find object
        object_info = self.find_object(object_name)
        if not object_info:
            print(f"❌ Object '{object_name}' not found!")
            return None
        
        object_path = Path(object_info['path'])
        
        # Create zip file
        zip_file = Path(export_path) / f"{object_info['safe_name']}.zip"
        
        shutil.make_archive(str(zip_file).replace('.zip', ''), 'zip', str(object_path))
        
        print(f"✅ Exported: {zip_file}")
        return zip_file
    
    def import_object(self, zip_path):
        """
        Import an object from a zip file.
        
        Args:
            zip_path: Path to the zip file to import
        
        Returns:
            Path to the imported object folder
        """
        zip_file = Path(zip_path)
        
        if not zip_file.exists():
            print(f"❌ Zip file not found: {zip_file}")
            return None
        
        # Extract zip file
        extract_path = self.base_dir / zip_file.stem
        shutil.unpack_archive(str(zip_file), str(extract_path))
        
        # Load metadata
        metadata_file = extract_path / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Add to catalog
            if metadata['safe_name'] not in [obj['safe_name'] for obj in self.catalog['objects']]:
                self.catalog['objects'].append({
                    'name': metadata['name'],
                    'safe_name': metadata['safe_name'],
                    'category': metadata['category'],
                    'path': str(extract_path),
                    'created': metadata['created']
                })
                self.save_catalog()
        
        print(f"✅ Imported: {extract_path}")
        return extract_path


def interactive_object_manager():
    """Interactive CLI for object management."""
    manager = ObjectManager()
    
    print("\n" + "="*70)
    print("3D SCAN OBJECT MANAGER")
    print("="*70)
    print("\nOptions:")
    print("  1. Create new object")
    print("  2. List all objects")
    print("  3. Start capture session")
    print("  4. List sessions for object")
    print("  5. Verify and fix catalog")
    print("  6. Export object")
    print("  7. Import object")
    print("  8. Exit")
    print("="*70)
    
    while True:
        choice = input("\nEnter choice (1-8): ").strip()
        
        if choice == '1':
            name = input("Object name: ").strip()
            category = input("Category (vehicles/furniture/electronics/other): ").strip() or "other"
            manager.create_object_folder(name, category)
        
        elif choice == '2':
            category = input("Filter by category (leave empty for all): ").strip() or None
            manager.list_objects(category)
        
        elif choice == '3':
            name = input("Object name: ").strip()
            session_name = input("Session name (leave empty for auto): ").strip() or None
            manager.start_capture_session(name, session_name)
        
        elif choice == '4':
            name = input("Object name: ").strip()
            manager.list_sessions(name)
        
        elif choice == '5':
            manager.verify_and_fix_catalog()
        
        elif choice == '6':
            name = input("Object name to export: ").strip()
            export_path = input("Export path (directory): ").strip()
            manager.export_object(name, export_path)
        
        elif choice == '7':
            zip_path = input("Zip file path to import: ").strip()
            manager.import_object(zip_path)
        
        elif choice == '8':
            print("\n✅ Goodbye!")
            break
        
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    interactive_object_manager()