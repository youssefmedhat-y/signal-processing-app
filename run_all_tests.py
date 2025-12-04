"""
Comprehensive Test Runner for Task 6 - Time Domain Operations
Runs all official test files from the task folders
"""
import sys
import os
import subprocess

def run_test(test_name, test_dir, test_file):
    """Run a single test file"""
    print("\n" + "="*60)
    print(f"Running: {test_name}")
    print("="*60)
    
    test_path = os.path.join(test_dir, test_file)
    
    if not os.path.exists(test_path):
        print(f"❌ Test file not found: {test_path}")
        return False
    
    try:
        # Change to test directory and run
        original_dir = os.getcwd()
        os.chdir(test_dir)
        
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        os.chdir(original_dir)
        
        if result.returncode == 0:
            if result.stdout:
                print(result.stdout)
            print(f"✅ {test_name} PASSED")
            return True
        else:
            print(f"❌ {test_name} FAILED")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {test_name} TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {test_name} ERROR: {e}")
        return False


def main():
    print("="*60)
    print("TASK 6 - TIME DOMAIN OPERATIONS TEST SUITE")
    print("="*60)
    print("Running all official test files...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    task_dir = os.path.join(base_dir, "task7", "task_and_files")
    
    results = {}
    
    # Test 1: Convolution
    conv_dir = os.path.join(task_dir, "TestCases", "Convolution")
    results["Convolution"] = run_test("Convolution Test", conv_dir, "ConvTest.py")
    
    # Test 2: Correlation (Point 1 - same length)
    corr_dir = os.path.join(task_dir, "Point1 Correlation")
    results["Correlation"] = run_test("Correlation Test", corr_dir, "CompareSignal.py")
    
    # Test 3: Shift and Fold
    shift_dir = os.path.join(task_dir, "TestCases", "Shifting and Folding")
    results["Shift/Fold"] = run_test("Shift and Fold Test", shift_dir, "Shift_Fold_Signal.py")
    
    #   
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20s} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    # Also run custom test suite
    print("\n" + "="*60)
    print("Running Custom Test Suite (test_time_domain.py)")
    print("="*60)
    
    test_suite_path = os.path.join(base_dir, "test_time_domain.py")
    if os.path.exists(test_suite_path):
        # Try to use Anaconda Python if available, otherwise use sys.executable
        python_exe = "D:/apps/anaconda/python.exe" if os.path.exists("D:/apps/anaconda/python.exe") else sys.executable
        result = subprocess.run([python_exe, test_suite_path], cwd=base_dir)
        if result.returncode == 0:
            print("\n✅ Custom test suite completed")
        else:
            print("\n❌ Custom test suite had errors")
    
    print("\n" + "="*60)
    print("All testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
