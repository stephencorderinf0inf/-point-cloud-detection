"""
Translation String Extractor
=============================
Extracts all translatable strings from the scanner for translation.
Generates .pot (template) and .po (language) files.

Usage:
    python extract_translations.py
"""

import re
from pathlib import Path

def extract_strings_from_file(file_path):
    """
    Extract translatable strings from a Python file.
    Looks for _("...") patterns.
    
    Args:
        file_path: Path to Python file
        
    Returns:
        List of (string, line_number) tuples
    """
    strings = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            # Match _("string") or _('string')
            matches = re.findall(r'_\(["\'](.+?)["\']\)', line)
            
            for match in matches:
                strings.append((match, line_num))
    
    return strings

def generate_pot_template(strings, output_file):
    """
    Generate .pot template file.
    
    Args:
        strings: List of (string, file, line_number) tuples
        output_file: Path to output .pot file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write('# 3D Laser Scanner Translation Template\n')
        f.write('# Copyright (C) 2025\n')
        f.write('# This file is distributed under the same license as the scanner package.\n')
        f.write('#\n')
        f.write('msgid ""\n')
        f.write('msgstr ""\n')
        f.write('"Content-Type: text/plain; charset=UTF-8\\n"\n')
        f.write('"Language: en\\n"\n')
        f.write('\n')
        
        # Strings
        seen = set()
        for string, file_path, line_num in strings:
            if string not in seen:
                seen.add(string)
                f.write(f'#: {file_path}:{line_num}\n')
                f.write(f'msgid "{string}"\n')
                f.write('msgstr ""\n')
                f.write('\n')
    
    print(f"✓ Generated template: {output_file}")
    print(f"  {len(seen)} unique strings")

def create_po_file_for_language(pot_file, lang_code, lang_name):
    """
    Create a .po file for a specific language from template.
    
    Args:
        pot_file: Path to .pot template
        lang_code: Language code (e.g., 'es')
        lang_name: Language name (e.g., 'Spanish')
    """
    locale_dir = Path(__file__).parent / 'locales' / lang_code / 'LC_MESSAGES'
    locale_dir.mkdir(parents=True, exist_ok=True)
    
    po_file = locale_dir / 'scanner.po'
    
    # Copy template and update header
    with open(pot_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update language in header
    content = content.replace('"Language: en\\n"', f'"Language: {lang_code}\\n"')
    
    with open(po_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Created {lang_name} translation file: {po_file}")
    print(f"  Edit this file to add {lang_name} translations")

def compile_po_to_mo():
    """
    Compile .po files to .mo binary files.
    Requires: pip install polib
    """
    try:
        import polib
    except ImportError:
        print("\n⚠️  polib not installed. Install with:")
        print("   pip install polib")
        return
    
    locales_dir = Path(__file__).parent / 'locales'
    
    for po_file in locales_dir.rglob('*.po'):
        mo_file = po_file.with_suffix('.mo')
        
        try:
            po = polib.pofile(str(po_file))
            po.save_as_mofile(str(mo_file))
            print(f"✓ Compiled: {mo_file.name}")
        except Exception as e:
            print(f"❌ Failed to compile {po_file.name}: {e}")

if __name__ == "__main__":
    print("=" * 70)
    print("Translation String Extractor")
    print("=" * 70)
    
    # Files to scan
    scanner_dir = Path(__file__).parent
    files_to_scan = [
        'laser_3d_scanner_advanced.py',
        'calibration_helper.py',
        # Add more files as needed
    ]
    
    all_strings = []
    
    print("\nScanning files...")
    for file_name in files_to_scan:
        file_path = scanner_dir / file_name
        if file_path.exists():
            strings = extract_strings_from_file(file_path)
            for string, line_num in strings:
                all_strings.append((string, file_name, line_num))
            print(f"  {file_name}: {len(strings)} strings")
    
    if not all_strings:
        print("\n⚠️  No translatable strings found!")
        print("   Make sure you've wrapped strings with _(\"...\")")
        exit(1)
    
    # Generate .pot template
    pot_file = scanner_dir / 'locales' / 'scanner.pot'
    pot_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating translation template...")
    generate_pot_template(all_strings, pot_file)
    
    # Create .po files for each language
    print("\nCreating language-specific files...")
    languages = {
        'es': 'Spanish',
        # 'zh': 'Chinese (Simplified)',
        # 'de': 'German',
        # 'ja': 'Japanese',
    }
    
    for lang_code, lang_name in languages.items():
        create_po_file_for_language(pot_file, lang_code, lang_name)
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("\n1. TRANSLATE strings:")
    print("   • Open: locales/es/LC_MESSAGES/scanner.po")
    print("   • Edit msgstr \"\" fields with Spanish translations")
    print("   • Use a PO editor like Poedit (https://poedit.net/)")
    
    print("\n2. COMPILE translations:")
    print("   • Install: pip install polib")
    print("   • Run this script again to compile .po -> .mo")
    
    print("\n3. TEST translations:")
    print("   • Run: python i18n_manager.py")
    print("   • Or set language in scanner")
    
    print("\n" + "=" * 70)
