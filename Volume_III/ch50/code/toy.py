# The eight-car toy table (shared by hand.py and the listings)
import pandas as pd
cars = pd.DataFrame({
    "colour": ["red", "red", "red", "blue", "blue",
               "white", "white", "white"],
    "fuel": ["petrol", "diesel", "petrol", "petrol", "diesel",
             "diesel", "petrol", "petrol"],
    "age": [1, 2, 4, 3, 6, 5, 9, 13.],
    "price": [15, 12, 9, 10, 8, 9, 6, 3.]})
