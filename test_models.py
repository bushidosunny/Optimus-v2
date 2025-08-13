#!/usr/bin/env python3
"""
Test script to verify the 2025 model lineup
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_handler import LLMHandler
import logging

logging.basicConfig(level=logging.INFO)

def test_models():
    """Test the 2025 model availability"""
    print("Testing 2025 Model Lineup...")
    print("=" * 50)
    
    # Initialize LLM handler
    llm = LLMHandler()
    
    # Get available models
    available_models = llm.get_available_models()
    
    print(f"\n📋 Available Models ({len(available_models)} total):")
    print("-" * 30)
    
    # Group by provider
    providers = {}
    for model in available_models:
        provider = model.split(':')[0]
        if provider not in providers:
            providers[provider] = []
        providers[provider].append(model)
    
    for provider, models in providers.items():
        print(f"\n🏢 {provider.upper()}:")
        for model in sorted(models):
            model_name = model.split(':')[1]
            cost = llm.estimate_cost(model, 1000)
            cost_str = f"${cost:.4f}" if cost > 0 else "Free"
            print(f"   • {model_name} - {cost_str}/1K tokens")
    
    # Test 2025 flagship models
    print(f"\n🚀 2025 Flagship Models:")
    print("-" * 30)
    
    flagship_models = [
        "openai:gpt-5",
        "anthropic:claude-4-opus", 
        "anthropic:claude-4-sonnet",
        "xai:grok-4"
    ]
    
    for model in flagship_models:
        if model in available_models:
            cost = llm.estimate_cost(model, 1000)
            print(f"   ✅ {model} - ${cost:.4f}/1K tokens")
        else:
            print(f"   ❌ {model} - Not available (check API key)")
    
    # Get model info for GPT-5
    if "openai:gpt-5" in available_models:
        print(f"\n📊 GPT-5 Model Info:")
        info = llm.get_model_info("openai:gpt-5")
        for key, value in info.items():
            print(f"   {key}: {value}")
    
    print(f"\n✅ Model testing complete!")
    print(f"💰 Total cost for 1M tokens across all models: ${sum(llm.estimate_cost(m, 1000000) for m in available_models):.2f}")

if __name__ == "__main__":
    test_models()