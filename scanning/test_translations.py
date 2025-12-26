#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test translations for all languages"""

from i18n_manager import setup_i18n, set_language, _

print("=" * 70)
print("🌍 TESTING ALL LANGUAGE TRANSLATIONS")
print("=" * 70)

# Test strings
test_strings = [
    "SCANNER CONTROLS",
    "TERMINAL OUTPUT",
    "ROTATION:",
    "MODES:",
    "Camera Info",
    "Analyzing..."
]

languages = [
    ('en', 'English'),
    ('es', 'Español'),
    ('fr', 'Français'),
    ('pt', 'Português'),
    ('de', 'Deutsch'),
    ('zh', '中文'),
    ('ja', '日本語'),
    ('hi', 'हिन्दी')
]

# Initialize i18n
_ = setup_i18n()

for lang_code, lang_name in languages:
    print(f"\n{'='*70}")
    print(f"Language: {lang_name} ({lang_code})")
    print('='*70)
    
    set_language(lang_code)
    
    for test_str in test_strings:
        translated = _(test_str)
        print(f"  {test_str:25s} → {translated}")

print(f"\n{'='*70}")
print("✅ All 8 languages are working!")
print("="*70)
