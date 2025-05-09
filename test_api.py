#!/usr/bin/env python3
from src.alphavantage import getData, api_keys

def main():
    """
    Test the Alpha Vantage API with proper error handling.
    This script will attempt to fetch a small amount of data to test the error handling.
    """
    print("Testing Alpha Vantage API with proper error handling")
    
    # Initialize the getData class with SPY symbol
    getter = getData(symbol='SPY')
    
    # Test with a very short date range (3 days) to minimize API calls
    success = getter.run('2023-01-04', '2023-01-06', 'test_output.json')
    
    if success:
        print("Test completed successfully. Check the test_output.json file.")
    else:
        print("Test failed. No data was collected.")

if __name__ == "__main__":
    main() 