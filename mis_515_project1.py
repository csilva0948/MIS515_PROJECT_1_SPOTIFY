# Import Lib
import pandas as pd

# Find the CSV file path
file_path = "/Users/carlos/Documents/spotify-2023.csv" # file_path = "/Users/carlos/Documents/data.csv" 

# Read the CSV file
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Print the data
print(df)

## Cleaning Phase ##
missing_values = df.isnull().sum()

# Display the count of missing values for each column
print(missing_values)

# Optional: Display columns with missing values only
missing_columns = missing_values[missing_values > 0]
print("Columns with missing values:")
print(missing_columns)

df_cleaned = df.dropna(axis=0, how='any')

# Display the cleaned DataFrame
print("\nDataFrame after removing all rows with missing values:")
print(df_cleaned)

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv('/Users/carlos/Documents/cleaned_spotify_2023.csv', index=False)

# Check the data types of each column
print(df_cleaned.dtypes)

# Convert the object column "Streams" to numeric one to prevent errors in visualization
df_cleaned['streams'] = pd.to_numeric(df_cleaned['streams'], errors='coerce')





## Data Visualization Phase ## 

### Distribution of Songs by Key in Major and Minor Modes ###

# Count the number of songs in each key
key_distribution = df_cleaned['key'].value_counts()

# Display the distribution
print("Distribution of songs by key:")
print(key_distribution)


# Count the number of songs in each key for Major mode
major_key_distribution = df_cleaned[df_cleaned['mode'] == 'Major']['key'].value_counts()

# Count the number of songs in each key for Minor mode
minor_key_distribution = df_cleaned[df_cleaned['mode'] == 'Minor']['key'].value_counts()

# Combine the two distributions into a single DataFrame for plotting
key_distribution_df = pd.DataFrame({
    'Major': major_key_distribution,
    'Minor': minor_key_distribution
}).fillna(0)  

# Display the distribution
print("Distribution of songs by key in Major and Minor modes:")
print(key_distribution_df)

import matplotlib.pyplot as plt

# Create a bar plot for the distribution of songs by key in Major and Minor modes
key_distribution_df.plot(kind='bar', figsize=(12, 6))
plt.title('Distribution of Songs by Key in Major and Minor Modes')
plt.xlabel('Musical Key')
plt.ylabel('Number of Songs')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.grid(axis='y')

# Show the plot
plt.legend(title='Mode')
plt.show()




### Mean Danceability by Released Year (2018 to 2023) ###

# Calculate the mean of audio features by released year
mean_audio_features_by_year = df_cleaned.groupby('released_year').agg({
    'bpm': 'mean',
    'danceability_%': 'mean',
    'valence_%': 'mean',
    'energy_%': 'mean',
    'acousticness_%': 'mean',
    'liveness_%': 'mean',
    'speechiness_%': 'mean'
})


# Round all columns to 0 decimal places (whole numbers) and convert to integers
mean_audio_features_rounded = mean_audio_features_by_year.round(0).astype(int)

# Display the result rounded
print(mean_audio_features_rounded)

# Filter the DataFrame for years between 2018 and 2023
filtered_df = df_cleaned[(df_cleaned['released_year'] >= 2018) & (df_cleaned['released_year'] <= 2023)]

# Group by 'released_year' and calculate the mean danceability
danceability_by_year = filtered_df.groupby('released_year')['danceability_%'].mean()

# Display the mean danceability by year
print("Mean Danceability by Year (2018 to 2023):")
print(danceability_by_year)

# Visualize the mean danceability over the years
plt.figure(figsize=(12, 6))
danceability_by_year.plot(kind='line', marker='o', color='blue')
plt.title('Mean Danceability by Released Year (2018 to 2023)')
plt.xlabel('Released Year')
plt.ylabel('Mean Danceability (%)')
plt.xticks(danceability_by_year.index)  # Set x-ticks to be the years
plt.grid(axis='y')

# Show the plot
plt.show()


### BPM Distribution by Streaming Ranking ###

# DataFrame "bpm_distribution" that has mean, min, max, and standard deviation of BPM values for each category in "in_spotify_charts"
bpm_distribution = df_cleaned.groupby('in_spotify_charts')['bpm'].agg(['mean', 'min', 'max', 'std']).reset_index()

# Display the distribution of BPM
print("Distribution of BPM per Streaming Ranking (Spotify Charts):")
print(bpm_distribution)

import pandas as pd

# Sort the DataFrame by streams in descending order
df_sorted = df_cleaned.sort_values(by='streams', ascending=False)

# Create a new column to categorize the stream rankings
def categorize_streams(row, rank_order):
    if row['streams'] in rank_order[:10]:
        return 'Top 10'
    elif row['streams'] in rank_order[10:50]:
        return 'Top 50'
    elif row['streams'] in rank_order[50:100]:
        return 'Top 100'
    elif row['streams'] in rank_order[100:500]:
        return 'Top 500'
    else:
        return 'Below Top 500'

# Get the top stream counts for categorization
rank_order = df_sorted['streams'].nlargest(500).tolist()  # Get top 500 stream counts

# Apply the function to create a new column for stream ranking
df_sorted['stream_ranking'] = df_sorted.apply(lambda row: categorize_streams(row, rank_order), axis=1)

# Filter to include only the desired ranking groups
desired_rankings = ['Top 500', 'Top 100', 'Top 50', 'Top 10']
top_streams = df_sorted[df_sorted['stream_ranking'].isin(desired_rankings)]

# Group by the 'stream_ranking' column and calculate statistics for BPM
bpm_stats = top_streams.groupby('stream_ranking')['bpm'].agg(['mean', 'min', 'max', 'std']).reset_index()

# Display the distribution of BPM
print("Distribution of BPM per Streaming Ranking:")
print(bpm_stats)

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

# Create a Figure
plt.figure(figsize=(12, 6))

# Define x-axis values for the streaming rankings
x = np.arange(len(bpm_stats['stream_ranking']))

# Each plt.plot call plots a different statistical measure of BPM (mean, min, max, std deviation) against the x-axis ('stream_ranking')
plt.plot(x, bpm_stats['mean'], marker='o', label='Mean BPM', color='blue', linewidth=2)
plt.plot(x, bpm_stats['min'], marker='o', label='Min BPM', color='green', linewidth=2)
plt.plot(x, bpm_stats['max'], marker='o', label='Max BPM', color='orange', linewidth=2)
plt.plot(x, bpm_stats['std'], marker='o', label='Standard Deviation BPM', color='red', linewidth=2)


# marker='o' specifies that data points will be marked with circles.
# label='...' assigns a label to each line for the legend.
# color='...' sets the line color.
# linewidth=2 specifies the thickness of the lines.


# Add titles and labels
plt.title('BPM Distribution by Streaming Ranking', fontsize=16)
plt.xlabel('Streaming Ranking', fontsize=14)
plt.ylabel('BPM Values', fontsize=14)
# These lines add a title to the plot and label the x-axis and y-axis, with specified font sizes for better visibility.

plt.xticks(x, bpm_stats['stream_ranking'])  # This sets the x-ticks to correspond to the stream_ranking values, making the x-axis more informative.

plt.legend() # Displays the legend on the plot, which helps differentiate between the various lines plotted.

plt.grid() # Adds a grid to the plot, making it easier to read values from the graph.


#These lines add a title to the plot and label the x-axis and y-axis, with specified font sizes for better visibility.

# Adding captions
for i, value in enumerate(bpm_stats['mean']):
    plt.text(x[i], value, f'{value:.1f}', ha='center', va='bottom', fontsize=9, color='blue')

# Adding annotations for min BPM values
for i, value in enumerate(bpm_stats['min']):
    plt.text(x[i], value, f'{value}', ha='center', va='bottom', fontsize=9, color='green')

# Adding annotations for max BPM values
for i, value in enumerate(bpm_stats['max']):
    plt.text(x[i], value, f'{value}', ha='center', va='bottom', fontsize=9, color='orange')

# Adding annotations for standard deviation BPM values
for i, value in enumerate(bpm_stats['std']):
    plt.text(x[i], value, f'{value:.1f}', ha='center', va='bottom', fontsize=9, color='red')



# plt.text() places text at the specified (x, value) coordinates.
# ha='center' and va='bottom' control the horizontal and vertical alignment of the text
# Different colors to match the lines


# Show the plot
plt.tight_layout() # Adjusts the padding of the plot to ensure that all elements fit well within the figure and are not overlapping.
plt.show()

