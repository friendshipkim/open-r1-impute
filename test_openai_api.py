#!/usr/bin/env python3
"""
Test script to verify OpenAI API key is working with the new API format.
"""

import os
import openai

def test_openai_api():
    """Test if the OpenAI API key is working."""
    
    # Get API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable is not set.")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    try:
        # Initialize client with new API format
        client = openai.OpenAI(api_key=api_key)
        
        # Test with a simple request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'Hello, API is working!'"}
            ],
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ API key is working! Response: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing API key: {e}")
        return False

if __name__ == "__main__":
    test_openai_api() 