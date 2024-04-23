import pandas as pd
from langdetect import detect
from googletrans import Translator
from tqdm import tqdm
import re
from multiprocessing import Pool, cpu_count

# Initialize translator
translator = Translator()

# Function to clean text
def clean_text(text):
    # Remove emojis, special characters, and HTML tags
    cleaned_text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)  # Remove special characters
    cleaned_text = re.sub(r'[\uD800-\uDBFF][\uDC00-\uDFFF]', '', cleaned_text)  # Remove emojis
    return cleaned_text

# Function to translate text to English if not in English
def translate_to_english(text):
    if not text.strip():  # Skip translation if text is empty or whitespace
        return text
    
    try:
        # Detect the language of the text
        detected_lang = detect(text)
        
        # Translate to English if not already in English
        if detected_lang != 'es':
            translation = translator.translate(text, src=detected_lang, dest='es')
            return translation.text
        return text
    except Exception as e:
        print(f"Error translating '{text}': {e}")
        return text

if __name__ == '__main__':
    df = pd.read_csv('Data/challenge - dataset.csv')
    
    df['about'] = df['about'].apply(clean_text)
    df['keywords'] = df['keywords'].apply(clean_text)
    
    # Set up multiprocessing pool
    num_processes = max(1, cpu_count() // 2)  # Reduce number of processes to avoid rate limits
    with Pool(num_processes) as pool:
        df['about'] = list(tqdm(pool.imap(translate_to_english, df['about']), total=len(df), desc='Translating About'))
        df['keywords'] = list(tqdm(pool.imap(translate_to_english, df['keywords']), total=len(df), desc='Translating Keywords'))
    
    df.to_csv('Data/translated_dataset_es.csv', index=False)
