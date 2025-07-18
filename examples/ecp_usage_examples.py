"""
ECP Two-Stage Visual Reasoning Example Usage

This example demonstrates how to use the newly implemented ECP (Explicit Chaining of Perception)
two-stage reasoning functionality in Ovis for enhanced visual understanding and reasoning.
"""

# Example 1: Using standard OvisRunner with ECP enabled
def example_ovis_runner_with_ecp():
    """
    Example showing how to use the standard OvisRunner with ECP two-stage reasoning enabled.
    """
    from PIL import Image
    from ovis.serve.runner import RunnerArguments, OvisRunner
    
    # Create runner arguments with ECP enabled
    args = RunnerArguments(
        model_path='AIDC-AI/Ovis2-8B',  # Replace with actual model path
        enable_ecp_two_stage=True,      # Enable ECP two-stage reasoning
        ecp_perception_max_new_tokens=256,
        ecp_perception_temperature=0.7,
        max_new_tokens=512
    )
    
    # Create runner
    runner = OvisRunner(args)
    
    # Load image and prepare query
    image = Image.open('path/to/your/image.jpg')
    query = "What is the main action happening in this image?"
    
    # Run two-stage inference
    response = runner.run([image, query])
    
    # Access results
    print("=== ECP Two-Stage Results ===")
    print(f"Original Query: {response['original_query']}")
    print(f"Perception Stage: {response['perception_result']}")
    print(f"Final Answer: {response['output']}")
    print(f"Two-Stage Mode: {response['ecp_two_stage']}")
    
    return response


# Example 2: Using dedicated ECPRunner
def example_ecp_runner():
    """
    Example showing how to use the dedicated ECPRunner for advanced ECP functionality.
    """
    from PIL import Image
    from ovis.serve.ecp_runner import ECPRunnerArguments, ECPRunner
    
    # Create ECP-specific arguments
    args = ECPRunnerArguments(
        model_path='AIDC-AI/Ovis2-8B',
        enable_two_stage=True,
        perception_max_new_tokens=256,
        perception_temperature=0.7,
        max_new_tokens=512,
        perception_prompt_template=(
            "Analyze this image carefully and describe: "
            "1. Main objects and their properties "
            "2. Actions or activities taking place "
            "3. Spatial relationships between objects "
            "4. Any text or symbols visible "
            "5. Overall scene context and setting"
        )
    )
    
    # Create ECP runner
    runner = ECPRunner(args)
    
    # Load image and prepare query
    image = Image.open('path/to/your/image.jpg')
    query = "Based on what you see, what might happen next in this scene?"
    
    # Run ECP two-stage inference
    response = runner.run([image, query])
    
    print("=== Advanced ECP Results ===")
    print(f"Detailed Perception: {response['perception_result']}")
    print(f"Reasoning Analysis: {response['output']}")
    
    return response


# Example 3: Convenience function usage
def example_convenience_function():
    """
    Example using the convenience function for quick ECP setup.
    """
    from PIL import Image
    from ovis.serve.ecp_runner import create_ecp_runner
    
    # Quick ECP runner creation
    runner = create_ecp_runner(
        model_path='AIDC-AI/Ovis2-8B',
        enable_two_stage=True,
        perception_max_new_tokens=200,
        max_new_tokens=400
    )
    
    # Multiple images example
    image1 = Image.open('image1.jpg')
    image2 = Image.open('image2.jpg')
    query = "Compare these two images and identify the differences."
    
    response = runner.run([image1, image2, query])
    return response


# Example 4: Switching between single-stage and two-stage modes
def example_mode_switching():
    """
    Example showing how to switch between single-stage and two-stage modes.
    """
    from PIL import Image
    from ovis.serve.runner import RunnerArguments, OvisRunner
    
    # Create runner with ECP capabilities
    args = RunnerArguments(
        model_path='AIDC-AI/Ovis2-8B',
        enable_ecp_two_stage=False,  # Start with single-stage
        max_new_tokens=512
    )
    
    runner = OvisRunner(args)
    image = Image.open('path/to/your/image.jpg')
    query = "Describe this image."
    
    # Single-stage inference (standard behavior)
    single_stage_response = runner.run_single_stage([image, query])
    print("Single-stage response:", single_stage_response['output'])
    
    # Enable two-stage mode
    runner.enable_ecp_two_stage = True
    
    # Two-stage inference
    two_stage_response = runner.run([image, query])
    print("Two-stage perception:", two_stage_response['perception_result'])
    print("Two-stage reasoning:", two_stage_response['output'])
    
    return single_stage_response, two_stage_response


# Example 5: Custom perception prompts
def example_custom_perception_prompts():
    """
    Example showing how to use custom perception prompts for specific use cases.
    """
    from ovis.serve.ecp_runner import ECPRunnerArguments, ECPRunner
    
    # Custom prompt for medical image analysis
    medical_prompt = (
        "As a medical imaging expert, analyze this image and describe: "
        "1. Anatomical structures visible "
        "2. Any abnormalities or noteworthy features "
        "3. Image quality and technical aspects "
        "4. Relevant clinical observations"
    )
    
    args = ECPRunnerArguments(
        model_path='AIDC-AI/Ovis2-8B',
        perception_prompt_template=medical_prompt,
        enable_two_stage=True
    )
    
    runner = ECPRunner(args)
    # Would use with medical images
    
    return runner


# Example 6: Batch processing with ECP
def example_batch_processing():
    """
    Example showing how to process multiple images with ECP reasoning.
    """
    from PIL import Image
    from ovis.serve.ecp_runner import create_ecp_runner
    
    # Create runner
    runner = create_ecp_runner(
        model_path='AIDC-AI/Ovis2-8B',
        enable_two_stage=True
    )
    
    # Process multiple images
    image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    results = []
    
    for image_path in image_paths:
        image = Image.open(image_path)
        query = "What is the main subject of this image?"
        
        response = runner.run([image, query])
        results.append({
            'image_path': image_path,
            'perception': response['perception_result'],
            'answer': response['output']
        })
    
    return results


if __name__ == "__main__":
    print("ECP Two-Stage Visual Reasoning Examples")
    print("=" * 50)
    print()
    print("This file contains examples of how to use the ECP implementation.")
    print("To run these examples, you need:")
    print("1. A trained Ovis model (e.g., AIDC-AI/Ovis2-8B)")
    print("2. Images to analyze")
    print("3. PyTorch and other dependencies installed")
    print()
    print("Key features:")
    print("- Two-stage reasoning: perception → reasoning")
    print("- Backward compatibility with standard Ovis")
    print("- Customizable perception prompts")
    print("- Multiple usage patterns supported")
    print()
    print("See the function examples above for detailed usage patterns.")