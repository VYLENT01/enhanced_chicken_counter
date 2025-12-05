# Contributing to Enhanced Chicken Counter

Thank you for your interest in contributing to the Enhanced Chicken Counter project! This document provides guidelines and information for contributors.

## 🎯 Project Overview

The Enhanced Chicken Counter is an AI-powered system for automated chicken counting using YOLOv8 detection and ByteTrack tracking algorithms. Our goal is to achieve 97.4%+ precision in real-world poultry farm environments.

## 🤝 How to Contribute

### Types of Contributions Welcome

- **Bug reports and fixes**
- **Performance improvements**
- **New features and enhancements**
- **Documentation improvements**
- **Test coverage expansion**
- **Agricultural domain expertise**
- **Hardware optimization**
- **Mobile/edge deployment**

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/yourusername/enhanced-chicken-counter.git
cd enhanced-chicken-counter

# Add upstream remote
git remote add upstream https://github.com/originalowner/enhanced-chicken-counter.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run setup
python install.py

# Run tests to ensure everything works
python -m pytest tests/
```

### 3. Create Feature Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

## 📝 Development Guidelines

### Code Style

We follow Python PEP 8 standards with some project-specific conventions:

```python
# Use type hints
def count_chickens(frame: np.ndarray, confidence: float = 0.25) -> int:
    """Count chickens in the given frame.
    
    Args:
        frame: Input video frame
        confidence: Detection confidence threshold
        
    Returns:
        Number of chickens detected
    """
    pass

# Use descriptive variable names
detection_results = model.detect(frame)
chicken_tracks = tracker.update(detection_results)

# Keep functions focused and small
def calculate_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """Calculate Intersection over Union of two bounding boxes."""
    # Implementation here
    pass
```

### Code Formatting

We use automated formatting tools:

```bash
# Install formatting tools
pip install black isort flake8 mypy

# Format code
black src/ utils/ tests/
isort src/ utils/ tests/

# Check code quality
flake8 src/ utils/ tests/
mypy src/
```

### Documentation

- **Docstrings**: Use Google-style docstrings for all functions and classes
- **Comments**: Explain complex algorithms and business logic
- **README updates**: Update README.md for new features
- **API documentation**: Document public APIs thoroughly

### Testing

#### Test Structure

```
tests/
├── unit/                   # Unit tests
│   ├── test_model_manager.py
│   ├── test_tracking.py
│   └── test_detection.py
├── integration/            # Integration tests
│   ├── test_full_pipeline.py
│   └── test_api_integration.py
├── performance/            # Performance tests
│   ├── test_fps_benchmarks.py
│   └── test_memory_usage.py
└── fixtures/               # Test data
    ├── sample_video.mp4
    └── test_images/
```

#### Writing Tests

```python
import pytest
import numpy as np
from src.model_manager import ModelManager

class TestModelManager:
    @pytest.fixture
    def model_manager(self):
        return ModelManager()
    
    def test_model_loading(self, model_manager):
        """Test that model loads successfully."""
        assert model_manager.initialize()
        assert model_manager.model is not None
    
    def test_detection_with_dummy_data(self, model_manager):
        """Test detection with synthetic data."""
        dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = model_manager.detect(dummy_frame)
        
        assert result is not None
        assert hasattr(result, 'detections')
```

#### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/

# Run with coverage
python -m pytest --cov=src tests/

# Run performance tests
python -m pytest tests/performance/ -v
```

## 🔬 Performance Considerations

### Optimization Guidelines

1. **Memory Efficiency**
   - Use generators for large data processing
   - Clean up resources properly
   - Monitor memory usage in long-running processes

2. **CPU/GPU Optimization**
   - Vectorize operations with NumPy
   - Use appropriate tensor operations
   - Implement proper batch processing

3. **Real-time Requirements**
   - Target 30+ FPS for real-time applications
   - Minimize latency in tracking pipeline
   - Optimize for specific hardware configurations

### Benchmarking

Before submitting performance improvements:

```bash
# Run benchmark suite
python utils/benchmark.py --duration 60

# Profile your changes
python -m cProfile -o profile.stats src/main_counter.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

## 🐛 Bug Reports

### Before Reporting

1. **Search existing issues** for similar problems
2. **Update to latest version** and test
3. **Run system validation**: `python utils/validation.py`
4. **Check logs** in `./logs/` directory

### Bug Report Template

```markdown
**Bug Description**
A clear description of the bug.

**Environment**
- OS: [e.g., Ubuntu 20.04, Windows 11]
- Python Version: [e.g., 3.9.7]
- GPU: [e.g., NVIDIA RTX 3080, CPU-only]
- Model: [e.g., yolov8s.pt]

**Reproduction Steps**
1. Configure system with...
2. Run command...
3. Observe error...

**Expected Behavior**
What should have happened.

**Actual Behavior**
What actually happened.

**Logs and Screenshots**
Attach relevant log files and screenshots.

**Additional Context**
Any other relevant information.
```

## ✨ Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature.

**Use Case**
Explain the agricultural/business need this addresses.

**Proposed Implementation**
Technical approach (if you have ideas).

**Alternatives Considered**
Other solutions you've considered.

**Additional Context**
Mockups, research papers, etc.
```

### Priority Features

We're particularly interested in contributions for:

- **Multi-species detection** (ducks, geese, turkeys)
- **Behavior analysis** (feeding, drinking, roosting)
- **Health monitoring** (lameness, stress indicators)
- **Environmental adaptation** (lighting, weather conditions)
- **Edge deployment** (Raspberry Pi, Jetson)
- **Mobile applications** (iOS, Android)
- **Farm management integration** (APIs, databases)

## 🔄 Pull Request Process

### 1. Preparation

```bash
# Ensure your branch is up to date
git fetch upstream
git rebase upstream/main

# Run quality checks
python -m pytest
black src/ utils/ tests/
flake8 src/
```

### 2. Commit Guidelines

Use conventional commit format:

```bash
# Feature additions
git commit -m "feat: add real-time behavior analysis"

# Bug fixes
git commit -m "fix: resolve tracking ID persistence issue"

# Performance improvements
git commit -m "perf: optimize model inference by 25%"

# Documentation
git commit -m "docs: update API documentation"

# Tests
git commit -m "test: add integration tests for tracking system"
```

### 3. Pull Request Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Test coverage

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance benchmarks run
- [ ] Manual testing completed

## Performance Impact
- FPS change: [e.g., +5 FPS improvement]
- Memory change: [e.g., -200MB reduction]
- Accuracy change: [e.g., +2% precision improvement]

## Screenshots/Videos
If applicable, add visual evidence of changes.

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] Performance impact assessed
```

### 4. Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Performance validation** for optimization PRs
4. **Agricultural domain review** for algorithm changes
5. **Final approval** and merge

## 🧪 Specialized Contributions

### Agricultural Domain Expertise

If you have poultry farming experience:

- **Validation**: Test with real farm environments
- **Requirements**: Define practical counting scenarios
- **Accuracy**: Validate against manual counts
- **Usability**: Feedback on farmer workflows

### Computer Vision Expertise

- **Algorithm improvements**: Better detection/tracking methods
- **Model optimization**: Quantization, pruning, distillation
- **Custom architectures**: Poultry-specific model designs
- **Data augmentation**: Farm-specific transformations

### Hardware/Deployment Expertise

- **Edge optimization**: ARM, mobile processors
- **Camera integration**: Hardware recommendations
- **Networking**: IoT connectivity solutions
- **Power efficiency**: Battery-powered deployments

## 📚 Resources for Contributors

### Technical Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [Computer Vision Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)
- [Agricultural AI Papers](https://www.nature.com/subjects/agricultural-engineering)

### Development Tools

- **IDEs**: VS Code, PyCharm
- **Debugging**: Python debugger, GPU profilers
- **Visualization**: Weights & Biases, TensorBoard
- **Testing**: pytest, hypothesis

### Communication

- **GitHub Issues**: Technical discussions
- **GitHub Discussions**: General questions
- **Code Reviews**: In-line PR comments

## 🏆 Recognition

Contributors will be recognized in:

- **README.md**: Major contributors section
- **Release notes**: Feature attributions
- **Documentation**: Technical contributions
- **Academic papers**: Research collaborations

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## ❓ Questions?

- **Technical questions**: Open a GitHub issue
- **Contribution ideas**: Start a GitHub discussion
- **Private inquiries**: Contact maintainers

---

**Thank you for contributing to making poultry farming more efficient and humane through AI technology! 🐔**