import pandas as pd
import pandas_datareader as pdr
from datetime import datetime

# Define the start and end dates
start_date = '1999-01-01'
end_date = '2023-12-31'

# Fetch the data
try:
    # Replace 'fred' with another data source if necessary
    df = pdr.get_data_fred('DEXUSEU', start=start_date, end=end_date)
    print(df)
except Exception as e:
    print("Error:", e)

print(df)