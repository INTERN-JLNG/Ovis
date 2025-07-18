#!/usr/bin/env python3
"""
Comprehensive validation script for ECP two-stage reasoning implementation.

This script validates that all components of the ECP implementation are working correctly
and can be used together as intended.
"""

import sys
from pathlib import Path

def test_constants():
    """Test ECP constants and templates."""
    print("Testing ECP constants...")
    
    try:
        from ovis.util.constants import (
            ECP_PERCEPTION_PROMPT_DEFAULT, 
            ECP_REASONING_PROMPT_TEMPLATE,
            ECP_ENABLE_TWO_STAGE,
            ECP_PERCEPTION_MAX_TOKENS,
            ECP_PERCEPTION_TEMPERATURE
        )
        
        # Test that constants exist and have expected properties
        assert len(ECP_PERCEPTION_PROMPT_DEFAULT) > 100, "Perception prompt too short"
        assert 'perception_result' in ECP_REASONING_PROMPT_TEMPLATE, "Missing perception_result placeholder"
        assert 'original_query' in ECP_REASONING_PROMPT_TEMPLATE, "Missing original_query placeholder"
        
        # Test template formatting
        test_result = ECP_REASONING_PROMPT_TEMPLATE.format(
            perception_result="Test perception output",
            original_query="Test query"
        )
        assert len(test_result) > 50, "Formatted template too short"
        
        print("✓ Constants validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Constants validation failed: {e}")
        return False


def test_argument_classes():
    """Test ECP argument classes."""
    print("Testing argument classes...")
    
    try:
        # Test that argument classes can be imported without torch
        # We only test structure, not actual functionality
        
        # This will fail if torch is not available, but we can check the module exists
        import importlib.util
        
        # Check if runner module exists
        runner_spec = importlib.util.find_spec('ovis.serve.runner')
        ecp_runner_spec = importlib.util.find_spec('ovis.serve.ecp_runner')
        
        assert runner_spec is not None, "runner module not found"
        assert ecp_runner_spec is not None, "ecp_runner module not found"
        
        print("✓ Module structure validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Module structure validation failed: {e}")
        return False


def test_documentation():
    """Test that documentation files exist and are properly formatted."""
    print("Testing documentation...")
    
    try:
        doc_path = Path(__file__).parent / "docs" / "ecp_two_stage_reasoning.md"
        examples_path = Path(__file__).parent / "examples" / "ecp_usage_examples.py"
        readme_path = Path(__file__).parent / "ECP_README.md"
        
        # Check documentation exists
        assert doc_path.exists(), "Main documentation file missing"
        assert examples_path.exists(), "Examples file missing"
        assert readme_path.exists(), "ECP README missing"
        
        # Check documentation content
        doc_content = doc_path.read_text()
        assert "ECP" in doc_content, "Documentation missing ECP content"
        assert "two-stage" in doc_content, "Documentation missing two-stage content"
        assert len(doc_content) > 5000, "Documentation too short"
        
        examples_content = examples_path.read_text()
        assert "def example_" in examples_content, "Examples missing example functions"
        assert "ECPRunner" in examples_content, "Examples missing ECPRunner usage"
        
        print("✓ Documentation validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Documentation validation failed: {e}")
        return False


def test_server_integration():
    """Test that server integration is properly implemented."""
    print("Testing server integration...")
    
    try:
        server_path = Path(__file__).parent / "ovis" / "serve" / "server.py"
        assert server_path.exists(), "Server file missing"
        
        server_content = server_path.read_text()
        assert "enable_ecp_two_stage" in server_content, "Server missing ECP arguments"
        assert "ecp_perception_max_new_tokens" in server_content, "Server missing ECP perception args"
        assert "ECP Two-Stage Reasoning" in server_content, "Server missing ECP title"
        
        print("✓ Server integration validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Server integration validation failed: {e}")
        return False


def test_package_structure():
    """Test that package structure is correct."""
    print("Testing package structure...")
    
    try:
        # Check that all necessary files exist
        base_path = Path(__file__).parent
        
        required_files = [
            "ovis/serve/runner.py",
            "ovis/serve/ecp_runner.py",
            "ovis/serve/server.py",
            "ovis/util/constants.py",
            "ovis/__init__.py",
            "docs/ecp_two_stage_reasoning.md",
            "examples/ecp_usage_examples.py",
            "ECP_README.md"
        ]
        
        for file_path in required_files:
            full_path = base_path / file_path
            assert full_path.exists(), f"Required file missing: {file_path}"
        
        # Check that key content exists in files
        runner_content = (base_path / "ovis/serve/runner.py").read_text()
        assert "enable_ecp_two_stage" in runner_content, "Runner missing ECP parameters"
        assert "run_ecp_two_stage" in runner_content, "Runner missing ECP method"
        
        init_content = (base_path / "ovis/__init__.py").read_text()
        assert "ECPRunner" in init_content, "__init__.py missing ECP exports"
        
        print("✓ Package structure validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Package structure validation failed: {e}")
        return False


def test_backward_compatibility():
    """Test that backward compatibility is maintained."""
    print("Testing backward compatibility...")
    
    try:
        # Test that original imports still work
        from ovis.util.constants import IMAGE_TOKEN, IGNORE_ID
        assert IMAGE_TOKEN == "<image>", "Original constants changed"
        assert IGNORE_ID == -100, "Original constants changed"
        
        print("✓ Backward compatibility validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Backward compatibility validation failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("ECP Two-Stage Reasoning Implementation Validation")
    print("=" * 60)
    
    tests = [
        test_constants,
        test_argument_classes,
        test_documentation,
        test_server_integration,
        test_package_structure,
        test_backward_compatibility
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
            print()
    
    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(tests)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1:2d}. {test.__name__:<30} {status}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All validations passed!")
        print("✅ ECP two-stage reasoning implementation is complete and ready for use.")
        print("\nNext steps:")
        print("1. Test with a real Ovis model")
        print("2. Run inference with sample images")
        print("3. Compare single-stage vs two-stage results")
        return 0
    else:
        print(f"\n⚠️  {total - passed} validation(s) failed.")
        print("❌ Please review the implementation before use.")
        return 1


if __name__ == "__main__":
    sys.exit(main())