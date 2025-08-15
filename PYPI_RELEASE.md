# 🚀 PyPI Release Guide for Lingo NLP Toolkit

This guide covers the complete process of preparing and releasing Lingo to PyPI (Python Package Index).

## **📋 Pre-Release Checklist**

### **1. Code Quality**
- [ ] All tests pass: `make test`
- [ ] Code is linted: `make lint`
- [ ] Code is formatted: `make format`
- [ ] Type checking passes: `mypy lingo/`
- [ ] Documentation is up to date
- [ ] Examples are working: `make examples`

### **2. Version Management**
- [ ] Update version in `lingo/__init__.py`
- [ ] Update `CHANGELOG.md` with new version
- [ ] Ensure all changes are documented
- [ ] Commit version bump: `git commit -m "Bump version to X.Y.Z"`

### **3. Package Configuration**
- [ ] Verify `setup.py` configuration
- [ ] Check `pyproject.toml` settings
- [ ] Validate `MANIFEST.in` includes all necessary files
- [ ] Test package building: `make build`

## **🔧 Building the Package**

### **Clean Build**
```bash
# Clean previous builds
make clean

# Build package
make build

# Verify package integrity
make build-check
```

### **Package Verification**
```bash
# Check package structure
twine check dist/*

# Test installation
python -m pip install dist/*.whl --force-reinstall

# Verify import works
python -c "import lingo; print(f'Lingo {lingo.__version__} imported successfully!')"
```

## **📦 Publishing to PyPI**

### **1. TestPyPI (Recommended First Step)**
```bash
# Publish to TestPyPI
make publish-test

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ lingo
```

### **2. PyPI (Production)**
```bash
# Publish to PyPI
make publish

# Verify on PyPI
# Visit: https://pypi.org/project/lingo-nlp-toolkit/#files
```

### **3. Full Release Process**
```bash
# Complete release workflow
make release
```

## **🔑 PyPI Authentication**

### **1. Create PyPI Account**
- Visit [PyPI](https://pypi.org/account/register/)
- Create account and verify email
- Enable two-factor authentication (recommended)

### **2. Generate API Token**
- Go to [PyPI Account Settings](https://pypi.org/manage/account/)
- Create API token with "Entire account" scope
- Copy the token (starts with `pypi-`)

### **3. Configure .pypirc**
```bash
# Copy template
cp .pypirc.template .pypirc

# Edit .pypirc with your tokens
nano .pypirc
```

**Example .pypirc:**
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-actual-token-here
repository = https://upload.pypi.org/legacy/

[testpypi]
username = __token__
password = pypi-your-test-token-here
repository = https://test.pypi.org/legacy/
```

**⚠️ Security Note:** Never commit `.pypirc` to version control!

## **📊 Package Distribution**

### **Supported Formats**
- **Source Distribution**: `.tar.gz` (recommended)
- **Wheel Distribution**: `.whl` (faster installation)

### **Installation Commands**
```bash
# Basic installation
pip install lingo-nlp-toolkit

# Full installation with all dependencies
pip install lingo-nlp-toolkit[full]

# Development installation
pip install lingo-nlp-toolkit[dev]

# GPU support
pip install lingo-nlp-toolkit[gpu]

# Documentation dependencies
pip install lingo-nlp-toolkit[docs]

# Testing dependencies
pip install lingo-nlp-toolkit[test]
```

## **🧪 Testing the Release**

### **1. TestPyPI Testing**
```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ lingo-nlp-toolkit

# Test basic functionality
python -c "from lingo import Pipeline; print('TestPyPI installation successful!')"

# Test CLI
lingo --help

# Test examples
python examples/showcase.py
```

### **2. PyPI Testing**
```bash
# Install from PyPI
pip install lingo-nlp-toolkit

# Test all functionality
make test
make examples
```

## **📈 Post-Release Tasks**

### **1. Verification**
- [ ] Package appears on PyPI
- [ ] Installation works: `pip install lingo-nlp-toolkit`
- [ ] Import works: `python -c "import lingo"`
- [ ] CLI works: `lingo --help`
- [ ] Examples run successfully

### **2. Documentation Updates**
- [ ] Update README with new version
- [ ] Update installation instructions
- [ ] Update examples if needed
- [ ] Update documentation sites

### **3. Announcements**
- [ ] GitHub release notes
- [ ] Social media announcements
- [ ] Community notifications
- [ ] Blog post (if applicable)

## **🚨 Troubleshooting**

### **Common Issues**

#### **1. Package Build Failures**
```bash
# Check for syntax errors
python -m py_compile lingo/

# Verify dependencies
pip install -r requirements.txt

# Clean and rebuild
make clean
make build
```

#### **2. Upload Failures**
```bash
# Check authentication
twine check dist/*

# Verify package format
python -m twine check dist/*

# Test with TestPyPI first
make publish-test
```

#### **3. Installation Issues**
```bash
# Check package integrity
twine check dist/*

# Verify package contents
tar -tzf dist/lingo-*.tar.gz
unzip -l dist/lingo-*.whl
```

### **Error Messages**

#### **"Package already exists"**
- Update version number
- Ensure version is unique

#### **"Invalid distribution"**
- Check package structure
- Verify MANIFEST.in
- Run `make clean` and rebuild

#### **"Authentication failed"**
- Check .pypirc configuration
- Verify API token
- Ensure token has correct permissions

## **📚 Additional Resources**

### **PyPI Documentation**
- [PyPI User Guide](https://packaging.python.org/guides/distributing-packages-using-setuptools/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Python Packaging Authority](https://www.pypa.io/)

### **Lingo-Specific Commands**
```bash
# Complete workflow
make release

# Test workflow
make test-publish

# Quality checks
make quality

# Performance testing
make benchmark
```

### **Useful Tools**
- **twine**: Package upload tool
- **build**: Modern package building
- **check-manifest**: Manifest validation
- **pyroma**: Package quality checking

## **🎯 Best Practices**

### **1. Version Management**
- Use semantic versioning (MAJOR.MINOR.PATCH)
- Update CHANGELOG.md for every release
- Tag releases in Git: `git tag v1.0.0`

### **2. Testing Strategy**
- Always test on TestPyPI first
- Test installation in clean environment
- Verify all examples work
- Run comprehensive test suite

### **3. Documentation**
- Keep README.md up to date
- Document breaking changes
- Provide migration guides
- Include usage examples

### **4. Security**
- Never commit credentials
- Use API tokens, not passwords
- Enable 2FA on PyPI account
- Regularly rotate tokens

---

**🚀 Happy Releasing! Your Lingo NLP Toolkit is ready for the world!**
