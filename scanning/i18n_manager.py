"""
Internationalization (i18n) Manager
====================================
Provides translation support for the 3D laser scanner.
Prepared for Microsoft Store multilingual publication.

Supported Languages:
- en_US (English - Default)
- es_ES (Spanish)
- zh_CN (Chinese Simplified) - Future
- de_DE (German) - Future
- ja_JP (Japanese) - Future
"""

import gettext
import locale
import os
from pathlib import Path

# Supported languages
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Español',
    'zh': '中文',
    'de': 'Deutsch',
    'ja': '日本語'
}

# Global translation function
_ = None  # Will be set by initialize()

class I18nManager:
    """Manages application localization."""
    
    def __init__(self, locale_dir=None, domain='scanner'):
        """
        Initialize i18n manager.
        
        Args:
            locale_dir: Directory containing .po/.mo files (default: ./locales)
            domain: Translation domain name (default: 'scanner')
        """
        if locale_dir is None:
            locale_dir = Path(__file__).parent / 'locales'
        
        self.locale_dir = Path(locale_dir)
        self.domain = domain
        self.current_lang = 'en'
        self._translation = None
        
    def detect_system_language(self):
        """
        Detect system language automatically.
        
        Returns:
            Language code (e.g., 'en', 'es', 'zh')
        """
        try:
            # Get system locale
            system_locale = locale.getdefaultlocale()[0]
            
            if system_locale:
                # Extract language code (e.g., 'en_US' -> 'en')
                lang_code = system_locale.split('_')[0].lower()
                
                # Return if supported
                if lang_code in SUPPORTED_LANGUAGES:
                    return lang_code
        except:
            pass
        
        # Default to English
        return 'en'
    
    def set_language(self, lang_code):
        """
        Set the active language.
        
        Args:
            lang_code: Language code ('en', 'es', 'zh', etc.)
        """
        if lang_code not in SUPPORTED_LANGUAGES:
            print(f"⚠️  Language '{lang_code}' not supported, using English")
            lang_code = 'en'
        
        self.current_lang = lang_code
        
        # Create locale directory if it doesn't exist
        self.locale_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load translation files
            if lang_code == 'en':
                # English is default - use NullTranslations
                self._translation = gettext.NullTranslations()
            else:
                # Load .mo file for the language
                mo_file = self.locale_dir / lang_code / 'LC_MESSAGES' / f'{self.domain}.mo'
                
                if mo_file.exists():
                    with open(mo_file, 'rb') as f:
                        self._translation = gettext.GNUTranslations(f)
                else:
                    print(f"⚠️  Translation file not found: {mo_file}")
                    print(f"   Using English fallback")
                    self._translation = gettext.NullTranslations()
        
        except Exception as e:
            print(f"⚠️  Failed to load translations: {e}")
            self._translation = gettext.NullTranslations()
    
    def get_text(self, message):
        """
        Get translated text.
        
        Args:
            message: English text to translate
            
        Returns:
            Translated text
        """
        if self._translation is None:
            return message
        
        return self._translation.gettext(message)
    
    def get_current_language(self):
        """Get current language code."""
        return self.current_lang
    
    def get_language_name(self):
        """Get current language name."""
        return SUPPORTED_LANGUAGES.get(self.current_lang, 'English')


# Global instance
_i18n_manager = None

def initialize(lang_code=None):
    """
    Initialize i18n system.
    
    Args:
        lang_code: Language code ('en', 'es', etc.) or None for auto-detect
        
    Returns:
        Translation function
    """
    global _i18n_manager, _
    
    _i18n_manager = I18nManager()
    
    # Auto-detect or use specified language
    if lang_code is None:
        lang_code = _i18n_manager.detect_system_language()
    
    _i18n_manager.set_language(lang_code)
    
    # Set global translation function
    _ = _i18n_manager.get_text
    
    return _

def set_language(lang_code):
    """Change active language."""
    global _
    if _i18n_manager:
        _i18n_manager.set_language(lang_code)
        _ = _i18n_manager.get_text

def get_language():
    """Get current language code."""
    if _i18n_manager:
        return _i18n_manager.get_current_language()
    return 'en'

def get_supported_languages():
    """Get list of supported languages."""
    return SUPPORTED_LANGUAGES.copy()


# Convenience function for quick setup
def setup_i18n(lang=None):
    """
    Quick setup for i18n.
    
    Usage:
        from i18n_manager import setup_i18n, _
        setup_i18n()  # Auto-detect
        print(_("Hello World"))
    
    Args:
        lang: Language code or None for auto-detect
    """
    return initialize(lang)


if __name__ == "__main__":
    # Test i18n system
    print("=" * 70)
    print("i18n Manager Test")
    print("=" * 70)
    
    # Initialize
    _ = setup_i18n()
    
    print(f"\nCurrent language: {get_language()}")
    print(f"Language name: {_i18n_manager.get_language_name()}")
    
    print("\nSupported languages:")
    for code, name in SUPPORTED_LANGUAGES.items():
        print(f"  {code}: {name}")
    
    print("\nTest translations:")
    test_strings = [
        "Camera calibration required",
        "Scanning in progress",
        "Point cloud generated",
        "Mesh export complete"
    ]
    
    for s in test_strings:
        print(f"  EN: {s}")
        print(f"  ->: {_(s)}")
