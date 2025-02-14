import torch
from transformers import DistilBertTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MovieDatasetWithFeatures(torch.utils.data.Dataset):
    def __init__(self, ratings_df, device):
        """
        ratings_df: DataFrame containing columns ['userId', 'itemId', 'like', 'age_group', 'gender', 'profession', genre columns]
        device: Device to load tensors on (e.g., 'cuda' or 'cpu')
        """
        self.device = device
        self.ratings_df = ratings_df
        self.genre_df = self.ratings_df.drop(columns=["userId", "itemId", "title", "rating", "timestamp", "gender", "profession", "age_group", "like"], errors='ignore')        
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.encoded_titles = self.tokenizer(
            ratings_df['title'].tolist(),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors=None,
        )
        
    def __len__(self):
        return len(self.ratings_df)

    def __getitem__(self, idx):
        row = self.ratings_df.iloc[idx]
        user_tensor = torch.tensor(row['userId'], dtype=torch.long).to(self.device)
        age_tensor = torch.tensor(row['age_group'], dtype=torch.long).to(self.device)
        gender_tensor = torch.tensor(row['gender'], dtype=torch.long).to(self.device)
        profession_tensor = torch.tensor(row['profession'], dtype=torch.long).to(self.device)
        movie_tensor = torch.tensor(row['itemId'], dtype=torch.long).to(self.device)
        title_ids = torch.tensor(self.encoded_titles['input_ids'][idx], dtype=torch.long).to(self.device)
        title_mask = torch.tensor(self.encoded_titles['attention_mask'][idx], dtype=torch.long).to(self.device)
        genre_tensor = torch.tensor(self.genre_df.iloc[idx].values, dtype=torch.float).to(self.device)
        like_tensor = torch.tensor(row['like'], dtype=torch.long).to(self.device)

        return user_tensor, age_tensor, gender_tensor, profession_tensor, movie_tensor, title_ids, title_mask, genre_tensor, like_tensor
