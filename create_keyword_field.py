import pandas as pd
from googletrans import Translator

# Load the dataset with the 'keywords' column
df = pd.read_csv('Data/filtered_dataset_es.csv')

# Initialize translator
translator = Translator()

# Function to translate keywords to English
def translate_to_english(keyword_list):
    translated_keywords = []
    for keyword in keyword_list:
        try:
            translation = translator.translate(keyword, src='auto', dest='en')
            translated_keywords.append(translation.text)
        except Exception as e:
            print(f"Translation failed for keyword: {keyword}")
            translated_keywords.append(keyword)
    return translated_keywords

# Translate non-English keywords
df['keywords'] = df['keywords'].apply(eval).apply(lambda x: translate_to_english(x))

# Flatten the list of lists and get unique values
unique_keywords = set([keyword for sublist in df['keywords'] for keyword in sublist])

# Convert the 'keywords' column into dummy variables
keywords_dummies = pd.DataFrame()
for keyword in unique_keywords:
    keywords_dummies[keyword] = df['keywords'].apply(lambda x: 1 if keyword in x else 0)

# Concatenate the dummy variables with the original DataFrame
df = pd.concat([df, keywords_dummies], axis=1)

# Drop the original 'keywords' column
df.drop(columns=['keywords'], inplace=True)

# Save the DataFrame to a new CSV file
df.to_csv('/home/tomd/Documenti/GitHub/Company-Env-Classificator/Data/translated_dataset.csv', index=False)

# Now you can use the new dummy variables in your model
print(df.head())
