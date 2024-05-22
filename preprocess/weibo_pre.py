#########################################################################################
# Processing the original Chinese Weibo dataset
#########################################################################################
import json
import os
import pandas as pd
import concurrent.futures

# Path to the original dataset
weibo_path = '../data/weibo/'
weibo_json_path='../data/weibo/json/'
# Path to save the cleaned data
weibo_output_path = '../data/weibo/weibo_clean/'

# Reads the original Weibo data JSON files, extracts post/comment mid, propagation parent node,
# post/comment text and publication time information, and outputs to the weibo_output_path.
# Each post is output as a separate CSV file.
def process_directory():
    weibo_files = os.listdir(weibo_json_path)
    # Use a thread pool to process the file conversion
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(json_to_csv, weibo_files)

# Extract content features and user features
def json_to_csv(weibo_files):
    file_name = weibo_files.split('.')[0]
    # bi_followers_count: Number of mutual followers, friends_count: Number of followings, followers_count: Number of followers
    # statuses_count: Number of posts, verified: Whether the user is verified, verified_type: Type of verification,
    # favourites_count: Number of favorites, user_created_at: User registration time, user_geo_enabled: Whether the user's location is enabled
    # reposts_count: Number of reposts, comments_count: Number of comments, attitudes_count: Number of likes
    key=['mid', 'parent', 'text', 't','bi_followers_count','friends_count','followers_count',
                                    'statuses_count','verified','verified_type','favourites_count','user_created_at','gender','user_geo_enabled',
                                    'reposts_count','comments_count','attitudes_count']
    # Create an empty DataFrame with predefined columns
    file_df = pd.DataFrame(columns=key)
    # Open the JSON file and load its content
    with open(weibo_json_path + weibo_files, 'r', errors="ignore", encoding="utf-8") as load_news:
        try:
            file = json.load(load_news)
            file_len = len(file)
            if file_len>500:    # Limit the maximum number of nodes to 2000
                file_len=500
            # Iterate through each post/comment in the JSON file
            for j in range(file_len):
                f=file[j]
                loc=[]
                for k in key:
                    v=f.get(k,-2)
                    if(k=='mid'and v==-2):
                        print(file_name)
                        break
                    loc.append(v)
                # Add the extracted information to the DataFrame
                file_df.loc[j] = loc
            # Save the DataFrame to a CSV file
            weibo_output = weibo_output_path + file_name + '.csv'
            file_df.to_csv(weibo_output, encoding='utf-8')
        except json.decoder.JSONDecodeError as e:
            print(e)
            # Log the unfinished file in case of JSON decode error
            with open(weibo_path + 'unfinished_files.txt', 'a', encoding='utf-8', newline='') as f:
                string = file_name + '\n'
                f.writelines(string)

# Output a txt file containing the labels for each post
def read_file_label():
    input_file = weibo_path+"Weibo.txt"
    output_file = weibo_path+"weibo_id_label.txt"
    # Open the input file and output file
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Read the input file line by line
        for line in infile:
            # Split the fields in each line
            fields = line.split()
            # Skip the line if there are less than 2 fields
            if len(fields) < 2:
                continue
            # Get the eip and label fields
            eip = fields[0].split(":")[1]
            label = fields[1].split(":")[1]
            string = eip + '\t' + label + '\n'
            # Write the eip and label to the output file
            outfile.write(string)

# Main function to process the directory and read file labels
if __name__ == '__main__':
    process_directory()
    read_file_label()



