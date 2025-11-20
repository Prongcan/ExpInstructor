#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the get_original_review_text tool function
"""

import json
import sys
import os
from typing import Annotated

# Add project path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def get_original_review_text(paper_id: Annotated[str, "Paper ID"], review_id: Annotated[str, "Review ID"]) -> str:
    """
    Retrieve original review text from iclr2024_simple.json based on paper_id and review_id
    
    Args:
        paper_id: The paper ID to search for
        review_id: The review ID to search for
    
    Returns:
        Original review text including summary, strengths, weakness, and suggestions
    """
    try:
        # Load the JSON file
        with open("data/ICLR/iclr2024_simple.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Search for matching paper_id and review_id
        for entry in data:
            if entry.get("paper_id") == paper_id:
                # Find matching review_id
                for review in entry.get("review_contents", []):
                    if review.get("review_id") == review_id:
                        content = review.get("content", {})
                        
                        result = f"Found original review text for Paper ID: {paper_id}, Review ID: {review_id}\n"
                        result += "=" * 80 + "\n"
                        result += "ORIGINAL REVIEW CONTENT:\n"
                        result += "-" * 50 + "\n"
                        
                        if content.get("strengths"):
                            result += f"STRENGTHS:\n{content['strengths']}\n\n"
                        if content.get("weakness"):
                            result += f"WEAKNESS:\n{content['weakness']}\n\n"
                        if content.get("suggestions"):
                            result += f"SUGGESTIONS:\n{content['suggestions']}\n\n"
                        
                        return result
        
        return f"No review found for Paper ID: {paper_id}, Review ID: {review_id}"
        
    except Exception as e:
        return f"Error retrieving review text: {str(e)}"


def test_function():
    """Test function"""
    print("=" * 80)
    print("Testing get_original_review_text function")
    print("=" * 80)
    
    # Test case 1: Use known existing paper_id and review_id
    print("\n[Test Case 1] Using real IDs from data file")
    print("-" * 80)
    
    # First read some real IDs from file
    try:
        with open("data/ICLR/iclr2024_simple.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get IDs and review_id of first few papers
        test_cases = []
        for i, entry in enumerate(data[:5]):  # Test first 5 papers
            paper_id = entry.get("paper_id")
            if entry.get("review_contents"):
                for review in entry.get("review_contents", [])[:2]:  # Take first 2 reviews per paper
                    review_id = review.get("review_id")
                    test_cases.append((paper_id, review_id))
                    if len(test_cases) >= 5:  # Maximum 5 test cases
                        break
            if len(test_cases) >= 5:
                break
        
        print(f"Found {len(test_cases)} test cases\n")
        
        # Execute tests
        success_count = 0
        fail_count = 0
        
        for idx, (paper_id, review_id) in enumerate(test_cases, 1):
            print(f"\nTest case {idx}:")
            print(f"  Paper ID: {paper_id}")
            print(f"  Review ID: {review_id}")
            
            result = get_original_review_text(paper_id, review_id)
            
            if result.startswith("Found"):
                print(f"  ✓ Successfully found review")
                print(f"  Result length: {len(result)} characters")
                # Display first 200 characters of result
                print(f"  Result preview:\n{result[:300]}...")
                success_count += 1
            elif result.startswith("No review found"):
                print(f"  ✗ Review not found: {result}")
                fail_count += 1
            elif result.startswith("Error"):
                print(f"  ✗ Error occurred: {result}")
                fail_count += 1
            else:
                print(f"  ? Unknown result: {result[:100]}")
                fail_count += 1
        
        print("\n" + "=" * 80)
        print("Test result statistics:")
        print(f"  Success: {success_count}/{len(test_cases)}")
        print(f"  Failed: {fail_count}/{len(test_cases)}")
        print("=" * 80)
        
        # Test case 2: Test non-existent ID
        print("\n[Test Case 2] Testing non-existent ID")
        print("-" * 80)
        fake_paper_id = "FAKE_PAPER_ID_12345"
        fake_review_id = "FAKE_REVIEW_ID_12345"
        result = get_original_review_text(fake_paper_id, fake_review_id)
        print(f"Paper ID: {fake_paper_id}")
        print(f"Review ID: {fake_review_id}")
        print(f"Result: {result}")
        
        # Test case 3: Check data file structure
        print("\n[Test Case 3] Checking data file structure")
        print("-" * 80)
        print(f"Total paper count: {len(data)}")
        paper_with_reviews = sum(1 for entry in data if entry.get("review_contents"))
        print(f"Papers with reviews: {paper_with_reviews}")
        
        # Count review numbers
        total_reviews = sum(len(entry.get("review_contents", [])) for entry in data)
        print(f"Total review count: {total_reviews}")
        
        # Check content fields
        sample_entry = data[0]
        if sample_entry.get("review_contents"):
            sample_review = sample_entry["review_contents"][0]
            content_keys = list(sample_review.get("content", {}).keys())
            print(f"Content fields: {content_keys}")
            
            # Check if there's a summary field but function doesn't return it
            if "summary" in content_keys:
                print("⚠️  Warning: 'summary' field exists in content, but function doesn't return it!")
        
    except FileNotFoundError:
        print("✗ Error: Data file not found")
        print(f"  Path: data/ICLR/iclr2024_simple.json")
    except json.JSONDecodeError as e:
        print(f"✗ Error: JSON parsing failed: {e}")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_function()

