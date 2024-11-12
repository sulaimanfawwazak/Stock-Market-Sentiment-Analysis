# Use "nlp" Conda environment
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

finviz_url = 'https://finviz.com/quote.ashx?t='

tickers = ['GOOG', 'NVDA', 'AAPL', 'MSFT']

news_tables = {}
for ticker in tickers:
  # Append the ticker name at the back of the finviz_url
  url = finviz_url + ticker

  # Create a request to the url
  request = Request(url=url, headers={'user-agent': 'my-app'})

  # Capture the response
  response = urlopen(request)
  
  # Parse the content of HTML using BeautifulSoup
  html = BeautifulSoup(response, 'html.parser')
  news_table = html.find(id='news-table')
  news_tables[ticker] = news_table


### Test codes ###
# amzn_data = news_tables['AMZN']
# amzn_rows = amzn_data.findAll('tr')
# 
# for idx, row in enumerate(amzn_rows):
#   title = row.a.text
#   timestamp = row.td.text.strip()
#   print(timestamp + ' ' + title)

# List to store the parsed data
parsed_data = []

# Iterate through the tables from news_tables
for ticker, news_table in news_tables.items():
  # Iterate through the rows of the tables, and find <tr> element
  for row in news_table.findAll('tr'):
    # Parse the title
    title = row.a.text
    # Remove the trailings
    title = title.strip()

    # Parse the date and time
    # Format: "Nov-10-24 09:05AM"
    date_and_time = row.td.text
    # Remove the trailings
    date_and_time = date_and_time.strip()
    # Split the time and date
    date_and_time = date_and_time.split(' ')

    # If the length of date_and_time is larger than 1
    if len(date_and_time) > 1:
      date = date_and_time[0]
      time = date_and_time[1]

    else:
      time = date_and_time[0]

    parsed_data.append([ticker, date, time, title])

df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

vader = SentimentIntensityAnalyzer()

f = lambda title: vader.polarity_scores(title)['compound']
df['compound'] = df['title'].apply(f)

df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

def convert_date(date_str):
  if date_str == 'Today':
    return pd.to_datetime('today')
  else:
    return date_str
  
def convert_time(time_str):
  return pd.to_datetime(time_str)
  
df['date'] = df['date'].apply(convert_date)
df['date'] = pd.to_datetime(df['date']).dt.date
df['time'] = df['time'].apply(convert_time).dt.time

work_df = df[['ticker', 'date', 'compound']]

plt.figure(figsize=(10, 8))

mean_df = work_df.groupby(['ticker', 'date']).mean()
mean_df = mean_df.unstack()
mean_df = mean_df.xs('compound', axis=1).transpose()

# Why do we need this `unstack()` and `xs()`
# mean_df after `groupby().mean()`:
# ticker | date       | compound
# AAPL	 \ 2024-11-10 |	0.1
# AAPL	 \ 2024-11-11 |	0.2
# GOOG	 \ 2024-11-10 |	-0.1
# GOOG	 \ 2024-11-11 |	0.3

# It is more desirable for the .plot() function to have:
# - Dates as the index (x-axis).
# - Tickers as separate columns (one for each line on the plot).
# - Sentiment scores as the values in these columns.

# So we need to change the mean_df to be:
# date	      | AAPL | GOOG
# 2024-11-10	| 0.1 |	-0.1
# 2024-11-11	| 0.2 |	0.3


mean_df.plot(kind='line')
plt.yticks(np.arange(-1, 1, 0.25))
plt.grid()
plt.show()

print(df.head(10))
print(mean_df.head(10))