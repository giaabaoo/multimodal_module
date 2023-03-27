import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


def preprocess_text(text):
    """
    Preprocesses text by removing punctuation, converting to lowercase, and removing stopwords
    """
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # convert to lowercase
    text = text.lower()

    # remove stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    text_tokens = nltk.word_tokenize(text)
    text = [word for word in text_tokens if not word in stop_words]

    # re-join text
    text = ' '.join(text)

    return text


if __name__ == '__main__':
    # specify path to transcript file
    transcript_file = '/home/dhgbao/Research_Monash/dataset/ccu-data/CCU_COMBINED_ANNOTATED_VIDEO/segmented_videos_text/M01000AJ7_0001.txt'

    # load transcript
    with open(transcript_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # extract start and end times of each turn-taking
    turn_taking_times = []
    for line in lines:
        if '-->' in line:
            start_time, end_time = line.split('-->')
            turn_taking_times.append((float(start_time), float(end_time)))

    # extract the text of each turn-taking
    turn_taking_text = []
    for i, line in enumerate(lines):
        if '-->' in line:
            continue
        text = line.strip()
        if text:
            turn_taking_text.append(preprocess_text(text))

    # perform sentiment analysis on each turn-taking
    sia = SentimentIntensityAnalyzer()
    moods = []
    for text in turn_taking_text:
        score = sia.polarity_scores(text)
        mood = score['compound']
        moods.append(mood)

    # create a DataFrame with time and mood values for each second
    df = pd.DataFrame(columns=['time', 'mood'])
    for i in range(len(turn_taking_times) - 1):
        start_time, end_time = turn_taking_times[i]
        mood = moods[i]
        time_range = end_time - start_time
        time_series = pd.Series(data=[start_time + j / 100 for j in range(int(time_range * 100) + 1)], name='time')
        mood_series = pd.Series(data=[mood] * len(time_series), name='mood')
        df_turn = pd.concat([time_series, mood_series], axis=1)
        df = pd.concat([df, df_turn])

    # plot the mood over time
    plt.plot(df['time'], df['mood'])
    plt.xlabel('Time (s)')
    plt.ylabel('Mood')
    plt.savefig("trans_predict_mood.png")
