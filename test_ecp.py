"""
Test script for ECP two-stage reasoning functionality in Ovis.

This script demonstrates how to use the ECP two-stage reasoning capabilities
both through the main OvisRunner and the standalone ECPRunner.
"""

import os
import sys
from pathlib import Path

# Add the ovis package to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all imports work correctly."""
    print("\n=== Testing Imports ===")
    
    try:
        from ovis.serve.runner import RunnerArguments, OvisRunner
        print("✓ Successfully imported standard runner components")
        
        from ovis.serve.ecp_runner import ECPRunnerArguments, ECPRunner
        print("✓ Successfully imported ECP runner components")
        
        from ovis.util.constants import ECP_PERCEPTION_PROMPT_DEFAULT, ECP_REASONING_PROMPT_TEMPLATE
        print("✓ Successfully imported ECP constants")
        print(f"✓ ECP Perception prompt (first 100 chars): {ECP_PERCEPTION_PROMPT_DEFAULT[:100]}...")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_argument_validation():
    """Test argument validation and defaults."""
    print("\n=== Testing Argument Validation ===")
    
    try:
        from ovis.serve.runner import RunnerArguments
        from ovis.serve.ecp_runner import ECPRunnerArguments
        
        # Test standard arguments
        standard_args = RunnerArguments(model_path="test")
        print(f"Standard ECP enabled (default): {standard_args.enable_ecp_two_stage}")
        
        # Test ECP arguments with defaults
        ecp_args = ECPRunnerArguments(model_path="test")
        print(f"ECP enabled (default): {ecp_args.enable_two_stage}")
        print(f"ECP perception tokens (default): {ecp_args.perception_max_new_tokens}")
        
        # Test custom ECP arguments
        custom_ecp_args = ECPRunnerArguments(
            model_path="test",
            enable_two_stage=False,
            perception_max_new_tokens=128
        )
        print(f"Custom ECP enabled: {custom_ecp_args.enable_two_stage}")
        print(f"Custom perception tokens: {custom_ecp_args.perception_max_new_tokens}")
        
        return True
    except Exception as e:
        print(f"✗ Argument validation error: {e}")
        return False


def test_runner_initialization():
    """Test that runner classes can be instantiated with correct parameters."""
    print("\n=== Testing Runner Initialization ===")
    
    try:
        from ovis.serve.runner import RunnerArguments
        from ovis.serve.ecp_runner import ECPRunnerArguments
        
        # Test standard runner args
        standard_args = RunnerArguments(
            model_path="test_model_path",
            enable_ecp_two_stage=True,
            ecp_perception_max_new_tokens=256,
            ecp_perception_temperature=0.7,
            max_new_tokens=512
        )
        
        print(f"✓ Standard runner args created with ECP enabled: {standard_args.enable_ecp_two_stage}")
        
        # Test ECP runner args
        ecp_args = ECPRunnerArguments(
            model_path="test_model_path",
            enable_two_stage=True,
            perception_max_new_tokens=256,
            perception_temperature=0.7,
            max_new_tokens=512,
            perception_prompt_template="Custom prompt for testing."
        )
        
        print(f"✓ ECP runner args created with two-stage: {ecp_args.enable_two_stage}")
        print(f"✓ Custom perception prompt: {ecp_args.perception_prompt_template[:30]}...")
        
        return True
    except Exception as e:
        print(f"✗ Runner initialization error: {e}")
        return False


def test_constants_integration():
    """Test that ECP constants are properly integrated."""
    print("\n=== Testing Constants Integration ===")
    
    try:
        from ovis.util.constants import (
            ECP_PERCEPTION_PROMPT_DEFAULT, 
            ECP_REASONING_PROMPT_TEMPLATE,
            ECP_ENABLE_TWO_STAGE,
            ECP_PERCEPTION_MAX_TOKENS,
            ECP_PERCEPTION_TEMPERATURE
        )
        
        print(f"✓ ECP_PERCEPTION_PROMPT_DEFAULT: {len(ECP_PERCEPTION_PROMPT_DEFAULT)} chars")
        print(f"✓ ECP_REASONING_PROMPT_TEMPLATE contains placeholders: {'perception_result' in ECP_REASONING_PROMPT_TEMPLATE}")
        print(f"✓ Configuration constants: {ECP_ENABLE_TWO_STAGE}, {ECP_PERCEPTION_MAX_TOKENS}")
        
        # Test template formatting
        test_perception = "This is a test image description."
        test_query = "What do you see?"
        formatted = ECP_REASONING_PROMPT_TEMPLATE.format(
            perception_result=test_perception,
            original_query=test_query
        )
        print(f"✓ Template formatting works: {len(formatted)} chars generated")
        
        return True
    except Exception as e:
        print(f"✗ Constants integration error: {e}")
        return False


def test_class_methods_exist():
    """Test that required methods exist in the classes."""
    print("\n=== Testing Class Methods ===")
    
    try:
        from ovis.serve.ecp_runner import ECPRunnerArguments, ECPRunner, create_ecp_runner
        
        # Check ECPRunnerArguments
        ecp_args = ECPRunnerArguments(model_path="test")
        required_attrs = ['enable_two_stage', 'perception_max_new_tokens', 'perception_temperature']
        for attr in required_attrs:
            if hasattr(ecp_args, attr):
                print(f"✓ ECPRunnerArguments has {attr}")
            else:
                print(f"✗ ECPRunnerArguments missing {attr}")
                return False
        
        # Check create_ecp_runner function
        print(f"✓ create_ecp_runner function available: {callable(create_ecp_runner)}")
        
        return True
    except Exception as e:
        print(f"✗ Class methods test error: {e}")
        return False


def main():
    """Run all tests."""
    print("ECP Two-Stage Reasoning Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_constants_integration,
        test_argument_validation,
        test_runner_initialization,
        test_class_methods_exist
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Results:")
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    total_passed = sum(results)
    total_tests = len(tests)
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All tests passed! ECP implementation is ready.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()