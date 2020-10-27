# Zenya Koprowksi
# FE 595 Homework 3
# https://towardsdatascience.com/merging-spreadsheets-with-python-append-f6de2d02e3f3

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from collections import Counter

class SentimentSorter:

    #https://www.simplifiedpython.net/python-get-files-in-directory/
    def dataAnalyzer(self, path):
        filenames = []
        files_in_path = path.iterdir()
        for item in files_in_path:
            if item.is_file():
                file = pd.read_csv(item)
                filenames.append(file)
        # Created one file with all of the data in it
        appended_df = pd.concat(filenames, ignore_index=True)
        appended_df.to_csv("AllCompanyNames.csv", index=False)

        # created the sid variable which will use the Sentiment Analyzer from the nltk package
        sid = SentimentIntensityAnalyzer()

        # Created a new column which extracted the copmound score of each of the copmanies
        appended_df["Scores"] = appended_df["Purpose"].apply(lambda x: sid.polarity_scores(x)['compound'])

        max = appended_df["Scores"].idxmax()
        max_name = appended_df.iloc[max]["Name"]

        min = appended_df["Scores"].idxmin()
        min_name = appended_df.iloc[min]["Name"]

        print("Best business idea:", max_name)
        print("Worst business idea:", min_name)

        # Loop through all of the purposes to find the amount of times that each word shoes up in the purpose list using
        # word tokenize from the nltk package
        cntr = Counter()
        for _, row in appended_df.iterrows():
            temp = Counter(word_tokenize(row["Purpose"]))
            cntr += temp

        # Print out the 10 most common words with the amount of times that they occur
        print(cntr.most_common(10))

def main():
    SentimentSorter.dataAnalyzer()

if __name__ == "__main__":
    main()
