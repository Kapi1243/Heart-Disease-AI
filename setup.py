import sys
import subprocess
from pathlib import Path


def check_python_version():
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"Python 3.8+ required. You have {version.major}.{version.minor}")
        return False
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_dependencies():
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install dependencies")
        return False


def verify_imports():
    print("\nVerifying key packages...")
    packages = [
        'numpy',
        'pandas',
        'sklearn',
        'xgboost',
        'imblearn',
        'matplotlib',
        'seaborn',
        'shap'
    ]
    
    failed = []
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed.append(package)
    
    return len(failed) == 0


def check_data_file():
    print("\nChecking data file...")
    data_file = Path("data") / "CVD_cleaned.csv"
    if data_file.exists():
        size_mb = data_file.stat().st_size / (1024 * 1024)
        print(f"Data file found ({size_mb:.1f} MB)")
        return True
    else:
        print(f"Data file not found at {data_file}")
        print("Please ensure CVD_cleaned.csv is in the data/ directory")
        return False


def create_directories():
    print("\nCreating project directories...")
    directories = [
        Path("models"),
        Path("reports"),
        Path("reports") / "eda",
        Path("reports") / "explainability",
        Path("logs")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"{directory}")
    
    return True


def run_quick_test():
    print("\nRunning quick test...")
    try:
        # Test imports from src modules
        sys.path.insert(0, str(Path("src")))
        from config import RANDOM_STATE
        from utils import setup_logging, set_seeds
        
        setup_logging('INFO')
        set_seeds(RANDOM_STATE)
        
        print("Project modules load correctly")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Heart Disease Prediction - Project Setup")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Install Dependencies", install_dependencies),
        ("Verify Imports", verify_imports),
        ("Data File", check_data_file),
        ("Create Directories", create_directories),
        ("Quick Test", run_quick_test)
    ]
    
    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("Setup Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(result for _, result in results)
    
    print("=" * 60)
    if all_passed:
        print("Setup complete! You're ready to run the pipeline.")
        print("\nNext steps:")
        print("  1. cd src")
        print("  2. python main.py")
        print("\nOr see QUICKSTART.md for more options.")
    else:
        print("Some checks failed. Please fix the issues above.")
        print("See README_new.md for troubleshooting.")
    print("=" * 60)


if __name__ == "__main__":
    main()
