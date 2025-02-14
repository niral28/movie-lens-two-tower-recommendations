import spacy
import pandas as pd
from typing import List, Dict
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder

class SpacyTitlePreprocessor:
    def __init__(self, model="en_core_web_sm"):
        # Load spaCy model
        self.nlp = spacy.load(model)
        
        # Custom patterns for movie-specific terms
        special_patterns = [
            {"label": "EDITION", "pattern": [{"LOWER": {"IN": ["director's", "directors", "director"]}, "OP": "?"}, 
                                          {"LOWER": "cut"}]},
            {"label": "EDITION", "pattern": [{"LOWER": "extended"}, {"LOWER": "edition", "OP": "?"}]},
            {"label": "EDITION", "pattern": [{"LOWER": "theatrical"}, {"LOWER": {"IN": ["cut", "version"]}, "OP": "?"}]},
            {"label": "YEAR", "pattern": [{"SHAPE": "dddd"}]}  # Match four digits
        ]
        
        # Add patterns to pipeline
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(special_patterns)

    def preprocess_title(self, title: str) -> Dict:
        """Process a single title using spaCy."""
        # Process the title
        doc = self.nlp(title)
        
        # Extract base title (excluding year and edition)
        edition_spans = [ent.text for ent in doc.ents if ent.label_ == "EDITION"]
        year_spans = [ent.text for ent in doc.ents if ent.label_ == "YEAR"]
        
        # Remove entities from title
        cleaned_title = title
        for span in edition_spans + year_spans:
            cleaned_title = cleaned_title.replace(span, "")
        
        # Clean up remaining punctuation and extra spaces
        cleaned_title = re.sub(r'[^\w\s-]', ' ', cleaned_title)
        cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()
        
        # Extract lemmatized tokens (excluding stopwords and punctuation)
        tokens = [token.lemma_.lower() for token in doc 
                 if not token.is_stop and not token.is_punct]
        
        return {
            'original_title': title,
            'cleaned_title': cleaned_title,
            'lemmatized_tokens': tokens,
            'edition': edition_spans[0] if edition_spans else None,
            'year': year_spans[0] if year_spans else None,
            'named_entities': [(ent.text, ent.label_) for ent in doc.ents],
            'pos_tags': [(token.text, token.pos_) for token in doc]
        }

    def process_titles(self, titles: List[str]) -> pd.DataFrame:
        """Process a list of titles and return as DataFrame."""
        processed_data = []
        
        for title in titles:
            title_data = self.preprocess_title(title)
            processed_data.append({
                'original_title': title_data['original_title'],
                'cleaned_title': title_data['cleaned_title'],
                'tokens': ' '.join(title_data['lemmatized_tokens']),
                'edition': title_data['edition'],
                'year': title_data['year'],
                'entities': title_data['named_entities'],
                'token_count': len(title_data['lemmatized_tokens'])
            })
        
        return pd.DataFrame(processed_data)

    def analyze_titles(self, titles: List[str]) -> Dict:
        """Generate analysis of the title dataset."""
        processed_df = self.process_titles(titles)
        
        # Collect all tokens
        all_tokens = ' '.join(processed_df['tokens']).split()
        token_freq = Counter(all_tokens)
        
        # Collect all named entities
        all_entities = [ent for entities in processed_df['entities'] 
                       for ent in entities]
        entity_types = Counter([ent[1] for ent in all_entities])
        
        return {
            'total_titles': len(titles),
            'unique_cleaned_titles': len(processed_df['cleaned_title'].unique()),
            'avg_token_count': processed_df['token_count'].mean(),
            'titles_with_year': processed_df['year'].notna().sum(),
            'titles_with_edition': processed_df['edition'].notna().sum(),
            'most_common_tokens': token_freq.most_common(10),
            'entity_type_distribution': dict(entity_types)
        }

def load_user_data(path:str)->pd.DataFrame:
  df_user = pd.read_csv(path, delimiter='|', header=None, names=['userId', 'age', 'gender', 'profession', 'zipcode'])

  # Define the bins and labels for age groups
  bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, float('inf')]
  labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60-65', '65-70', '70-75', '75+']
  # Create a new column 'age_group' by binning the 'age' column
  df_user['age_group'] = pd.cut(df_user['age'], bins=bins, labels=labels, right=False)
  return df_user

def load_data(path:str)->pd.DataFrame:
  df_data = pd.read_csv(path, delimiter='\t', header=None, names=['userId', 'itemId', 'rating', 'timestamp'])
  df_data['like'] = df_data['rating'].apply(lambda x: 1 if x>=4 else 0)
  return df_data

def load_item(path:str)->pd.DataFrame:
  df_item = pd.read_csv(path, delimiter='|', header=None, encoding='latin-1')
  df_item.columns = [
    'itemId', 'title', 'releaseDate', 'videoReleaseDate', 'imdbUrl', 'unknown', 'action', 'adventure', 'animation',
    'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'filmNoir', 'horror', 'musical', 'mystery', 'romance',
    'sciFi', 'thriller', 'war', 'western'
  ]
  df_item.drop(columns=['releaseDate', 'videoReleaseDate', 'imdbUrl'], inplace=True)
  title_preprocessor = SpacyTitlePreprocessor()
  title_analysis = title_preprocessor.process_titles(df_item['title'])
  df_item['title'] = title_analysis['cleaned_title']
  return df_item

def get_mergedDF(df_data, df_user, df_item):
  merged_df = df_data.merge(df_user, on='userId').merge(df_item, on='itemId')
  return merged_df

def encodeFeatures(df_user, le_age, le_job, le_gender):
  df_user['age_group'] = le_age.transform(df_user['age_group'])
  df_user['profession'] = le_job.transform(df_user['profession'])
  df_user['gender'] = le_gender.transform(df_user['gender'])
  df_user.drop(columns=['age', 'zipcode'], inplace=True)
