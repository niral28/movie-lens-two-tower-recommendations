import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from transformers import DistilBertModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def info_nce_loss(similarity_matrix, temperature=0.07):
    # Positive pairs are on diagonal
    batch_size = similarity_matrix.size(0)
    labels = torch.arange(batch_size, device=similarity_matrix.device)

    # Scale similarities
    similarity_matrix = similarity_matrix / temperature

    # InfoNCE loss: -log(exp(pos_score) / sum(exp(all_scores)))
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

class MovieTwoTowerModel(nn.Module):
  def __init__(self, num_users, num_movies, num_age_groups, num_professions, num_genres, embedding_dim):
    super(MovieTwoTowerModel, self).__init__()
    self.num_users = num_users
    self.num_movies = num_movies
    self.num_genres = num_genres
    self.num_age_groups = num_age_groups
    self.num_professions = num_professions
    self.embedding_dim = embedding_dim

    self.user_embedding = nn.Embedding(num_users, embedding_dim)
    self.age_embedding = nn.Embedding(num_age_groups, 4)
    self.profession_embedding = nn.Embedding(num_professions, 4)
    self.gender_embedding = nn.Embedding(2, 2)

    self.user_fc = nn.Sequential(
        nn.Linear(embedding_dim+4+4+2, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),

    )

    self.genre_fc = nn.Linear(num_genres, 16)
    self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
    self.movie_title_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
    # Freeze embedding layers
    for param in self.movie_title_encoder.parameters():
        param.requires_grad = False
    self.movie_title_projection = nn.Linear(768, 256)        
    self.movie_fc = nn.Sequential(
        
        nn.Linear(embedding_dim+256+16, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),

        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        
        nn.Linear(256, 128),
    )

    self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
    self.early_stopper = EarlyStopping(patience=5)
    self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
    self.temperature = 0.07
    self.train_loss = []
    self.val_loss = []

  def get_early_stopper(self):
    return self.early_stopper

  def get_metrics(self):
    return self.train_loss, self.val_loss

  def forward(self, user_input, age_input, gender_input, profession_input, movie_input, title_ids, title_mask, genre_feats, like=None):
    # User Tower
    user_embedding = self.user_embedding(user_input)
    age_embedding = self.age_embedding(age_input)
    profession_embedding = self.profession_embedding(profession_input)
    gender_embedding = self.gender_embedding(gender_input)
    user_feats = torch.cat([user_embedding, age_embedding, profession_embedding, gender_embedding], dim=1)
    user_feats = self.user_fc(user_feats)


    # Movie Tower
    movie_embedding = self.movie_embedding(movie_input)
    genre_embedding = self.genre_fc(genre_feats)
    title_embeddings = self.movie_title_encoder(
        title_ids, 
        attention_mask=title_mask
    ).last_hidden_state[:, 0, :] # CLS token
    title_embeddings = self.movie_title_projection(title_embeddings.float())
    movie_feats = torch.cat([movie_embedding, title_embeddings, genre_embedding], dim=1)
    movie_feats = self.movie_fc(movie_feats)
    user_feats = F.normalize(user_feats, p=2, dim=1)
    movie_feats = F.normalize(movie_feats, p=2, dim=1)
    # Compute Similarity Matrix
    similarity_matrix = torch.matmul(user_feats, movie_feats.transpose(0, 1))
    if like is not None:
      loss = info_nce_loss(similarity_matrix, temperature=self.temperature)
      return loss, similarity_matrix, user_feats, movie_feats
    else:
      return similarity_matrix, user_feats, movie_feats

  def user_tower(self, user_input, age_input, gender_input, profession_input):
    user_embedding = self.user_embedding(user_input)
    age_embedding = self.age_embedding(age_input)
    profession_embedding = self.profession_embedding(profession_input)
    gender_embedding = self.gender_embedding(gender_input)
    user_feats = torch.cat([user_embedding, age_embedding, profession_embedding, gender_embedding], dim=1)
    user_feats = self.user_fc(user_feats)
    user_feats = F.normalize(user_feats, p=2, dim=1)
    return user_feats

  def movie_tower(self, movie_input, title_ids, title_mask, genre_input):
    movie_embedding = self.movie_embedding(movie_input)
    genre_embedding = self.genre_fc(genre_input)
    title_embeddings = self.movie_title_encoder(
        title_ids, 
        attention_mask=title_mask
    ).last_hidden_state[:, 0, :]
    title_embeddings = self.movie_title_projection(title_embeddings.float())
    movie_feats = torch.cat([movie_embedding, title_embeddings, genre_embedding], dim=1)
    movie_feats = self.movie_fc(movie_feats)
    movie_feats = F.normalize(movie_feats, p=2, dim=1)
    return movie_feats

  def fit(self, epochs, train_dataloader, val_dataloader):
      for epoch in range(epochs):
          self.train()
          epoch_loss = 0.0
          batch_count = 0
          for batch in train_dataloader:
              user_input, age_input, gender_input, profession_input, movie_input, title_ids, title_mask, genre_feats, like = batch
              batch_count+=1
              # print(f'starting batch {batch_count}')
              loss, _ , _ , _ = self(user_input, age_input, gender_input, profession_input, movie_input, title_ids, title_mask, genre_feats, like)
              self.optimizer.zero_grad()
              loss.backward()
              self.optimizer.step()
              epoch_loss += loss.item()
              # print(f'finished batch {batch_count}')
          avg_train_loss = epoch_loss / len(train_dataloader)
          print(f"train step {epoch}: train loss {avg_train_loss:.4f}")
          self.train_loss.append(avg_train_loss)
          self.eval()
          val_loss = 0.0
          for batch in val_dataloader:
              user_input, age_input, gender_input, profession_input, movie_input, title_ids, title_mask, genre_feats, like = batch
              loss, _ , _ , _ = self(user_input, age_input, gender_input, profession_input, movie_input, title_ids, title_mask, genre_feats, like)
              val_loss += loss.item()
          avg_val_loss = val_loss / len(val_dataloader)
          print(f"val step {epoch}: val loss {avg_val_loss:.4f}")
          self.val_loss.append(avg_val_loss)
          self.scheduler.step(avg_val_loss)
          self.train()
          self.early_stopper(avg_val_loss, self)
          if self.early_stopper.early_stop:
              print("Early stopping")
              break
