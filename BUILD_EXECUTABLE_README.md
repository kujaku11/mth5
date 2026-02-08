# Building MTH5 Validator Standalone Executable

This directory contains scripts to build a standalone executable of the MTH5 validator using PyInstaller.

## Quick Start

### Windows
```cmd
python build_executable.py
```
Or simply double-click:
```cmd
build_executable.bat
```

### Linux/Mac
```bash
python build_executable.py
# or
chmod +x build_executable.sh
./build_executable.sh
```

## Build Methods

### Method 1: Python Script (Recommended)
**File**: `build_executable.py`

**Features**:
- Cross-platform (Windows, Linux, macOS)
- Automatic PyInstaller installation
- Progress messages and error handling
- Automatic testing of built executable

**Usage**:
```bash
python build_executable.py
```

### Method 2: Batch Script (Windows)
**File**: `build_executable.bat`

**Features**:
- Simple Windows batch file
- Quick build without Python script
- Can be run by double-clicking

**Usage**:
```cmd
build_executable.bat
```

### Method 3: Shell Script (Linux/Mac)
**File**: `build_executable.sh`

**Features**:
- Simple Unix shell script
- Quick build for Linux/macOS

**Usage**:
```bash
chmod +x build_executable.sh
./build_executable.sh
```

### Method 4: Spec File (Advanced)
**File**: `mth5-validator.spec`

**Features**:
- Fine-grained control over build
- Customizable for advanced users
- Can add icons, version info, etc.

**Usage**:
```bash
pyinstaller mth5-validator.spec
```

## Output

After building, you'll find:
```
dist/
├── mth5-validator.exe    (Windows)
└── mth5-validator        (Linux/Mac)
```

**Executable size**: ~50-100 MB (includes Python + all dependencies)

## Testing

Test the executable:
```bash
# Windows
dist\mth5-validator.exe --help
dist\mth5-validator.exe validate test.mth5

# Linux/Mac
./dist/mth5-validator --help
./dist/mth5-validator validate test.mth5
```

## Distribution

The executable in `dist/` is **completely standalone**:
- ✓ No Python required on target system
- ✓ No dependencies required
- ✓ Copy to any system and run

### Distributing the Executable

1. **Direct Copy**: Share the single executable file
2. **GitHub Release**: Attach as release artifact
3. **Website**: Host for download
4. **USB Drive**: Physical distribution

## Build Requirements

**Required**:
- Python 3.10+
- mth5 installed (`pip install mth5` or in development mode)

**Automatically Installed**:
- PyInstaller (installed automatically if not present)

## Troubleshooting

### Build Fails with Import Errors

Add missing hidden imports to the build script:
```python
'--hidden-import=missing_module',
```

### Executable is Too Large

The spec file excludes common unnecessary modules. To reduce further:

1. Edit `mth5-validator.spec`
2. Add more modules to `excludes` list:
```python
excludes=[
    'matplotlib',
    'tkinter',
    # Add more here
]
```

### Windows Defender Warning

**First run** may trigger Windows Defender (false positive).

**Solutions**:
- Whitelist the executable
- Code sign for production (requires certificate)
- Submit to Microsoft for analysis

### macOS Security Warning

Users may see "cannot be opened because it is from an unidentified developer"

**Solutions**:
- Right-click → Open (first time)
- System Preferences → Security → Allow
- Code sign and notarize for production

### Linux: Permission Denied

```bash
chmod +x dist/mth5-validator
./dist/mth5-validator
```

## Advanced Customization

### Adding an Icon

**Windows**:
1. Create `icon.ico` file
2. Edit spec file:
```python
exe = EXE(
    ...
    icon='icon.ico',
)
```

**macOS**:
1. Create `icon.icns` file
2. Edit spec file:
```python
exe = EXE(
    ...
    icon='icon.icns',
)
```

### Adding Version Information (Windows)

1. Create `version_info.txt`:
```python
VSVersionInfo(
    ffi=FixedFileInfo(
        filevers=(0, 6, 1, 0),
        prodvers=(0, 6, 1, 0),
    ),
    kids=[
        StringFileInfo([
            StringTable('040904B0', [
                StringStruct('CompanyName', 'USGS'),
                StringStruct('FileDescription', 'MTH5 File Validator'),
                StringStruct('FileVersion', '0.6.1'),
                StringStruct('ProductName', 'MTH5 Validator'),
            ])
        ]),
        VarFileInfo([VarStruct('Translation', [1033, 1200])])
    ]
)
```

2. Add to PyInstaller command:
```bash
--version-file=version_info.txt
```

### Multi-Platform Builds with GitHub Actions

Create `.github/workflows/build-executables.yml`:
```yaml
name: Build Executables

on:
  release:
    types: [created]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install pyinstaller
        pip install -e .
    
    - name: Build executable
      run: python build_executable.py
    
    - name: Upload artifact
      uses: actions/upload-artifact@v3
      with:
        name: mth5-validator-${{ runner.os }}
        path: dist/*
```

## Optimization Tips

### Reduce Size

1. Use `--exclude-module` for unnecessary packages
2. Enable UPX compression (included in spec file)
3. Use `--strip` to remove debug symbols (Linux/Mac)

### Improve Startup Speed

1. Use `--onefile` (already used)
2. Consider `--noupx` if startup is very slow
3. Test with `--bootloader-ignore-signals`

## Performance

**Build time**: 2-5 minutes (first build slower)
**Executable size**: 50-100 MB
**Startup time**: 1-3 seconds (cold start)
**Runtime performance**: Same as Python (no overhead)

## Comparison with Python Run

| Aspect | Python Script | Executable |
|--------|--------------|------------|
| **Requires Python** | Yes | No |
| **File size** | ~100KB | ~80MB |
| **Startup time** | <1s | 1-3s |
| **Performance** | Fast | Same |
| **Distribution** | Needs install | Copy & run |
| **Updates** | Easy (pip) | Rebuild needed |

## When to Use Executable

**Use executable when**:
- ✓ Distributing to non-Python users
- ✓ Need standalone tool
- ✓ Deploying to systems without Python
- ✓ Simplifying installation

**Use Python script when**:
- ✓ Users are Python developers
- ✓ Need easy updates (pip upgrade)
- ✓ Minimal file size required
- ✓ Part of larger Python workflow

## Support

For issues with:
- **Building**: Check PyInstaller docs
- **MTH5 Validator**: Open GitHub issue
- **Specific platforms**: See platform notes above

## Resources

- [PyInstaller Documentation](https://pyinstaller.org/)
- [MTH5 Documentation](https://mth5.readthedocs.io/)
- [MTH5 GitHub](https://github.com/kujaku11/mth5)
