# Internationalization (i18n) Guide
**3D Laser Scanner - Multi-Language Support**

## 📋 Overview
This scanner supports multiple languages for Microsoft Store publication.

**Currently Supported:**
- 🇺🇸 English (Default)
- 🇪🇸 Spanish (Ready for translation)

**Planned:**
- 🇨🇳 Chinese (Simplified)
- 🇩🇪 German
- 🇯🇵 Japanese

---

## 🚀 Quick Start for Developers

### 1. **Writing Translatable Code**

Wrap all user-facing strings with `_()`:

```python
from i18n_manager import setup_i18n, _

# Initialize at app start
_ = setup_i18n()  # Auto-detects system language

# Use _() for all UI text
print(_("Calibration loaded successfully"))
status = _("Scanning in progress")
```

**✅ DO wrap:**
- UI labels
- Status messages
- Error messages
- Button text
- Help text

**❌ DON'T wrap:**
- Log file messages
- Debug output
- Variable names
- File paths
- Technical identifiers

### 2. **Extract Strings for Translation**

Run the extraction script:

```bash
python extract_translations.py
```

This creates:
- `locales/scanner.pot` - Translation template
- `locales/es/LC_MESSAGES/scanner.po` - Spanish translation file

### 3. **Translate Strings**

Edit the `.po` file:

```po
#: laser_3d_scanner_advanced.py:123
msgid "Calibration loaded successfully"
msgstr "Calibración cargada exitosamente"

#: laser_3d_scanner_advanced.py:456
msgid "Scanning in progress"
msgstr "Escaneo en progreso"
```

**Tools:**
- [Poedit](https://poedit.net/) - Visual PO editor (Recommended)
- Any text editor

### 4. **Compile Translations**

```bash
pip install polib
python extract_translations.py
```

This generates `.mo` binary files used at runtime.

### 5. **Test Translations**

```python
from i18n_manager import setup_i18n, set_language, _

# Test Spanish
set_language('es')
print(_("Calibration loaded successfully"))
# Output: Calibración cargada exitosamente

# Switch back to English
set_language('en')
print(_("Calibration loaded successfully"))
# Output: Calibration loaded successfully
```

---

## 📁 Directory Structure

```
scanning/
├── i18n_manager.py           # Translation manager
├── extract_translations.py   # String extractor
├── locales/
│   ├── scanner.pot           # Translation template
│   ├── es/                   # Spanish
│   │   └── LC_MESSAGES/
│   │       ├── scanner.po    # Spanish translations (editable)
│   │       └── scanner.mo    # Compiled (auto-generated)
│   ├── zh/                   # Chinese (future)
│   │   └── LC_MESSAGES/
│   └── de/                   # German (future)
│       └── LC_MESSAGES/
└── laser_3d_scanner_advanced.py
```

---

## 🔄 Workflow

### **Adding New Features**

1. Write code with `_()` wrapped strings
2. Run `extract_translations.py`
3. Translate new strings in `.po` files
4. Compile with `extract_translations.py`

### **Adding New Language**

1. Edit `extract_translations.py`:
   ```python
   languages = {
       'es': 'Spanish',
       'zh': 'Chinese (Simplified)',  # Add this
   }
   ```

2. Run extraction script
3. Translate `locales/zh/LC_MESSAGES/scanner.po`
4. Compile translations

---

## 🎯 Best Practices

### **String Guidelines**

✅ **Good:**
```python
_("Calibration complete")
_("Error: Camera not found")
_(f"Scanned {count} points")  # Variables OK
```

❌ **Avoid:**
```python
_("calibration_complete")  # Use natural language
"Calibration " + _("complete")  # Don't split sentences
_("Click") + " " + _("here")  # Translators need context
```

### **Context for Translators**

Add comments for clarity:

```python
# Translator: This appears when mesh generation succeeds
_("Mesh export complete")

# Translator: Button label for starting calibration
_("Start Calibration")
```

### **Handling Plurals**

Use format strings:

```python
# Bad
if count == 1:
    msg = _("1 point captured")
else:
    msg = _(f"{count} points captured")

# Good
msg = _("{} point(s) captured").format(count)
```

---

## 🧪 Testing

### **Test All Languages**

```python
from i18n_manager import get_supported_languages, set_language, _

for lang_code, lang_name in get_supported_languages().items():
    set_language(lang_code)
    print(f"{lang_name}: {_('Calibration loaded')}")
```

### **Test on Different Windows Locales**

1. Change Windows language: Settings → Time & Language
2. Restart scanner
3. Verify auto-detection works

---

## 📦 Microsoft Store Preparation

### **Store Listing Requirements**

For each language, translate:
- App name
- Short description
- Full description
- Screenshots with localized UI
- Keywords

### **App Package Manifest**

Update `AppxManifest.xml`:

```xml
<Resources>
  <Resource Language="en-US"/>
  <Resource Language="es-ES"/>
  <Resource Language="zh-CN"/>
</Resources>
```

---

## 🛠️ Tools & Resources

**Translation Tools:**
- [Poedit](https://poedit.net/) - PO file editor
- [Microsoft Translator](https://www.bing.com/translator) - Initial translations
- [DeepL](https://www.deepl.com/translator) - High-quality machine translation

**Testing:**
- Windows Pseudo-Localization (detect hardcoded strings)
- Native speaker review (critical before Store launch)

**Documentation:**
- [Python gettext](https://docs.python.org/3/library/gettext.html)
- [GNU gettext manual](https://www.gnu.org/software/gettext/manual/)

---

## 📞 Support

**Found untranslated text?**
1. Wrap it with `_()`
2. Run `extract_translations.py`
3. Translate in `.po` files
4. Compile and test

**Translation issues?**
- Check `.po` file syntax
- Verify `.mo` files are compiled
- Test with `python i18n_manager.py`

---

## 🎓 Example Integration

See `laser_3d_scanner_advanced.py` for full implementation:

```python
# At module level
from i18n_manager import setup_i18n, _

# In main()
_ = setup_i18n()  # Auto-detect language

# Use throughout
print(_("Starting 3D scanner"))
cv2.putText(frame, _("Calibration Required"), ...)
```

---

**Last Updated:** December 2025  
**Version:** 1.0  
**For:** Microsoft Store Publication
