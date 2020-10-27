# Zenya Koprowksi
# FE 595 Homework 3
# https://towardsdatascience.com/merging-spreadsheets-with-python-append-f6de2d02e3f3

import pandas as pd
import os
from pathlib import Path
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from collections import Counter


# Received help from this website tp check file path https://www.pythondaddy.com/python/reading-all-files-in-a-directory-with-python/
def dataanalyzer():
    location = Path(r"C:\Users\zenya\PycharmProjects\Homework").rglob('*.csv')
    files = [x for x in location]
    col_names = ['Name', 'Purpose']
    dfs = [pd.read_csv(csv_file, names=col_names) for csv_file in files]
    # Created one file with all of the data in it
    appended_df = pd.concat(dfs, ignore_index=True)
    appended_df.to_csv("AllCompanyNames.csv", index=False)

    # created the sid variable which will use the Sentiment Analyzer from the nltk package
    sid = SentimentIntensityAnalyzer()

    # Created a new column which extracted the copmound score of each of the copmanies
    appended_df["Scores"] = appended_df["Purpose"].apply(lambda x: sid.polarity_scores(x)['compound'])

    # Finds the company name with the max score
    max = appended_df["Scores"].idxmax()
    max_name = appended_df.iloc[max]["Name"]

    # Finds the company name with the min score
    min = appended_df["Scores"].idxmin()
    min_name = appended_df.iloc[min]["Name"]

    # Print the conclusions
    print("Best business idea:", max_name)
    print("Worst business idea:", min_name)

    # Loop through all of the purposes to find the amount of times that each word shoes up in the purpose list using
    # word tokenize from the nltk package
    my_ctr = Counter()
    for _, row in appended_df.iterrows():
        temp = Counter(word_tokenize(row["Purpose"]))
        my_ctr += temp

    # Print out the 10 most common words with the amount of times that they occur
    print(my_ctr.most_common(10))


def main():
    dataanalyzer()


if __name__ == "__main__":
    main()
