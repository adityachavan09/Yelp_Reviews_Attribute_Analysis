from openai import OpenAI
import pandas as pd

client = OpenAI()

df0 = pd.read_csv('gpt_input_text.csv')

df = df0[df0['Food Quality'].isna()][['review_id', 'stars', 'text', 'Food Quality', 'Customer Service', 'Place', 'Menu_and_Pricing', 'Drinks', 'Time']]

df_test = df0

import re

# Adjusted function to instruct GPT on the format of the response
def get_gpt_response(review):
    prompt = (
        "You are an assistant skilled in analyzing Yelp reviews. "
        "Given a review, rate the aspects of Food Quality, Customer Service, Place (ambiance, cleanliness), "
        "Menu and Pricing (variety, price), Drinks (coffee, beverages, alcohol), and Time (Waiting time) on a scale from 1 (good), -1 (bad), or 0 "
        "(neutral or not mentioned). Return the ratings in the sequence of Food Quality; "
        "Customer Service; Place; Menu_and_Pricing; Drinks; Time separated by tabs. "
        "For example, a response should look like this: '-1\t1\t1\t-1\t0\t0'.\n\n"
        f"Review: {review}\n"
        "Ratings:"
    )
    
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt}
        ]
    )
    response = completion.choices[0].message.content
    
    # Extract the ratings from the response assuming they are in the correct order
    ratings = response.strip().split('\t')
    
    # Remove non-numeric characters from ratings
    ratings = [re.sub(r'[^\d-]', '', rating) for rating in ratings]
    
    return {
        'Food Quality': int(ratings[0]),
        'Customer Service': int(ratings[1]),
        'Place': int(ratings[2]),
        'Menu_and_Pricing': int(ratings[3]),
        'Drinks': int(ratings[4]),
        'Time': int(ratings[5])
    }

# Loop through the DataFrame
for index, row in df_test.iterrows():
    ratings = get_gpt_response(row['text'])

    # Update the DataFrame with the ratings
    df_test.at[index, 'Food Quality'] = ratings['Food Quality']
    df_test.at[index, 'Customer Service'] = ratings['Customer Service']
    df_test.at[index, 'Place'] = ratings['Place']
    df_test.at[index, 'Menu_and_Pricing'] = ratings['Menu_and_Pricing']
    df_test.at[index, 'Drinks'] = ratings['Drinks']
    df_test.at[index, 'Time'] = ratings['Time']

# Output the updated DataFrame
df_test.to_csv('gpt_output_labels.csv',index=False)
