# flight_api.py
from amadeus import Client, ResponseError
import config
import datetime

# Initialize the Amadeus client
amadeus = Client(
    client_id=config.API_KEY,
    client_secret=config.API_SECRET
)


def get_live_price(origin, destination, date_str):
    """
    Fetches the cheapest flight price from Amadeus API.
    Args:
        origin (str): IATA code (e.g., "DEL")
        destination (str): IATA code (e.g., "BOM")
        date_str (str): Date in "YYYY-MM-DD" format
    Returns:
        float: Price
        str: Currency (e.g., "INR")
    """
    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=date_str,
            adults=1,
            max=1  # We only need the cheapest one
        )

        if response.data:
            # Amadeus returns price as a string
            price_str = response.data[0]['price']['total']
            currency = response.data[0]['price']['currency']
            return float(price_str), currency
        else:
            return None, "No Data"

    except ResponseError as error:
        print(f"API Error: {error}")
        return None, "Error"