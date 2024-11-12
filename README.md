# Sentiment Analysis on Stock Market News
This repository contains Python code for the stock market sentiment analysis. 

## Project Overview
tickers = []
The sentiment analysis is done to the ticker-related news retrieved from [Finviz website](https://finviz.com). The Finviz website provides the news up to the last 6 days. The sentiment analysis is done using **NLTK** (Natural Language Toolkit), with the **VADER** (Valence Aware Dictionary and sEntiment Reasoner) lexicon. This analysis results in scores ranging between -1 (strongly negative) and 1 (strongly positive). The type of score that is used in this analysis is the compound score. The tickers that are tried to be analyzed are **AMZN**, **NVDA**, **AAPL**, **AMD**, and **INTC**.

## Tech Stack
* **NLTK** for natural language processing tasks, where in this case is determining sentiment analysis scores.
* **BeautifulSoup** for scraping the news headline from the Finviz website.
* **Pandas** for data handling and processing.
* **Matplotlib** for data visualization.
* **Numpy** for data manipulation.

## Dependencies Installation
`pip install -r requirements.txt`
