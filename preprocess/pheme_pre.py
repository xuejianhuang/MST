import os
import json
import pandas as pd

# Paths to raw and cleaned PHEME dataset directories
pheme_raw_path = '../data/pheme/all-rnr-annotated-threads/'
pheme_clean_path = '../data/pheme/pheme_clean/'

def read_json_file(file_path):
    """
    Reads a JSON file and returns its content as a dictionary.
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        dict or None: The content of the JSON file, or None if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file '{file_path}': {e}")
        return None

def process_event_data(event_path, label, label_dic,news_ids):
    """
    Processes the data for a specific event, extracting relevant information
    from source tweets and reactions, and saves it to a CSV file.
    Args:
        event_path (str): Path to the event directory.
        label (str): Label indicating whether the event is a rumor ('1') or non-rumor ('0').
        label_dic (dict): Dictionary to store news IDs and their corresponding labels.
        news_ids (list): List to store news IDs.
    """
    # Keys for the DataFrame columns
    df_key = ['mid', 'parent', 'text', 't', 'friends_count', 'followers_count', 'statuses_count',
              'verified', 'favourites_count', 'listed_count', 'user_created_at', 'following',
              'user_geo_enabled', 'protected', 'reposts_count', 'attitudes_count']

    # Keys for extracting content, user, and status information from the JSON
    content_key = ['id', 'in_reply_to_status_id', 'text', 'created_at']
    user_key = ['friends_count', 'followers_count', 'statuses_count', 'verified', 'favourites_count',
                'listed_count', 'created_at', 'following', 'geo_enabled', 'protected']
    stat_key = ['retweet_count', 'favorite_count']

    # Iterate through each news item in the event directory
    for news in os.listdir(event_path):
        if not news.startswith('._') and news != '.DS_Store': # Skip unwanted files
            news_ids.append(news)  # Add news ID to the list
            label_dic[news] = label  # Add label to the dictionary

            # Create an empty DataFrame with predefined columns
            news_df = pd.DataFrame(columns=df_key)

            # Path to the source tweet JSON file
            source_tweets_path = os.path.join(event_path, news, 'source-tweets', f'{news}.json')
            source_tweets_data = read_json_file(source_tweets_path)
            if source_tweets_data:  # If the source tweet data is successfully read
                # Extract relevant information and add it to the DataFrame
                loc_0 = [source_tweets_data.get(k, -1) for k in content_key]
                loc_0 += [source_tweets_data['user'].get(k, -1) for k in user_key]
                loc_0 += [source_tweets_data.get(k, -1) for k in stat_key]
                news_df.loc[0] = loc_0
            # Path to the reactions directory
            reactions_path = os.path.join(event_path, news, 'reactions')
            if os.path.exists(reactions_path):
                comment_list = os.listdir(reactions_path)

                # Iterate through each reaction (comment)
                for i, comment in enumerate(comment_list):
                    if not comment.startswith('._') and comment != '.DS_Store': # Skip unwanted files
                        comment_path = os.path.join(reactions_path, comment)
                        comment_data = read_json_file(comment_path)
                        if comment_data:  # If the comment data is successfully read
                            # Extract relevant information and add it to the DataFrame
                            loc = [comment_data.get(k, -1) for k in content_key]
                            loc += [comment_data['user'].get(k, -1) for k in user_key]
                            loc += [comment_data.get(k, -1) for k in stat_key]
                            news_df.loc[i + 1] = loc
            # Save the DataFrame to a CSV file
            pheme_output = os.path.join(pheme_clean_path, f'{news}.csv')
            news_df.to_csv(pheme_output, index=False)

def read_raw_data():
    """
    Reads the raw PHEME dataset, processes each event, and saves the cleaned data.
    Also creates a file mapping news IDs to their labels.
    """
    event_list = ['germanwings-crash-all-rnr-threads', 'charliehebdo-all-rnr-threads',
                  'sydneysiege-all-rnr-threads', 'ebola-essien-all-rnr-threads',
                  'gurlitt-all-rnr-threads', 'putinmissing-all-rnr-threads',
                  'ferguson-all-rnr-threads', 'ottawashooting-all-rnr-threads',
                  'prince-toronto-all-rnr-threads']

    label_dic = {}  # Dictionary to store news IDs and labels
    news_ids = []  # List to store news IDs
    # Iterate through each event in the list
    for event in event_list:
        event_path = os.path.join(pheme_raw_path, event)
        if not os.path.exists(event_path):
            continue

        # Process non-rumor data
        non_rumor_path = os.path.join(event_path, 'non-rumours')
        if os.path.exists(non_rumor_path):
            process_event_data(non_rumor_path, '0', label_dic,news_ids)
        # Process rumor data
        rumor_path = os.path.join(event_path, 'rumours')
        if os.path.exists(rumor_path):
            process_event_data(rumor_path, '1', label_dic,news_ids)

    # Output PHEME labels to a file
    with open('../data/pheme/pheme_id_label.txt', 'w', encoding='utf-8') as f:
        for news_id, label in label_dic.items():
            f.write(f"{news_id}\t{label}\n")

if __name__ == '__main__':
    read_raw_data()
