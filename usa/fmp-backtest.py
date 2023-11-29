import requests

# API Endpoint
url = "https://financialmodelingprep.com/api/v3/sp500_constituent"

# Your API Key (if required)
api_key = "9d3a358bc5165e8334c3f2f858e4c315"

# Parameters
params = {
    "apikey": api_key
}

response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Parse the response to JSON
    data = response.json()
    # Print the list of stocks
    for stock in data:
        print(stock)
else:
    print("Failed to fetch data: Status code", response.status_code)