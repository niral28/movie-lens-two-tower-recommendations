import faiss
import matplotlib.pyplot as plt 
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE

from .model import MovieTwoTowerModel
from .dataset import MovieDatasetWithFeatures, DistilBertTokenizer
from .preprocess import load_user_data, load_item, load_data, encodeFeatures, get_mergedDF, LabelEncoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR='data'

def plot_training_val_loss(model: MovieTwoTowerModel):
    train_losses, val_losses= model.get_metrics()
    epochs = range(len(train_losses))

    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(DATA_DIR, 'train_val.png'))

def plot_heatmaps(similarity_matrices:list):
    fig, ax = plt.subplots(nrows=len(similarity_matrices), figsize=(100, 100))
    for i, matrix in enumerate(similarity_matrices):
        plt.title('Matrix {}'.format(i))
        sns.heatmap(matrix.detach().cpu().numpy(), square=True, ax=ax[i])
    fig.savefig(os.path.join(DATA_DIR, f'heatmatp_batch.png'))

def evaluate(model: MovieDatasetWithFeatures, merged_df_test: pd.DataFrame):
    test_dataset = MovieDatasetWithFeatures(merged_df_test, device)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)
    similarity_matrices = []
    model.eval()
    test_loss = []
    for batch in test_dataloader:
        user_input, age_input, gender_input, profession_input, movie_input, title_ids, title_mask, genre_feats, like = batch
        loss, similarity_matrix, user_feats, movie_feats = model(user_input, age_input, gender_input, profession_input, movie_input, title_ids, title_mask, genre_feats, like)
        similarity_matrices.append(similarity_matrix)
        print(f"test loss {loss:.4f}")
        test_loss.append(loss.item())
    print(f"overall average test loss {np.mean(test_loss):.4f}")
    plot_heatmaps(similarity_matrices)

def manifold_plots(model: MovieDatasetWithFeatures, df_item: pd.DataFrame, df_user:pd.DataFrame):
    tsne = TSNE(random_state=1, n_iter=15000, metric="cosine")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoded_titles = tokenizer(
            df_item['title'].tolist(),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors=None,
        )
    title_ids = torch.tensor(encoded_titles['input_ids'], dtype=torch.long).to(device)
    title_mask = torch.tensor(encoded_titles['attention_mask'], dtype=torch.long).to(device)
    movie_embeddings = model.movie_tower(torch.tensor(df_item['itemId'].values, dtype=torch.long).to(device), title_ids, title_mask, torch.tensor(df_item.drop(columns=['itemId', 'title']).values, dtype=torch.float).to(device))
    movie_embeddings = movie_embeddings.detach().cpu().numpy()
    embs = tsne.fit_transform(movie_embeddings)
    df_item['x'] = embs[:, 0]
    df_item['y'] = embs[:, 1]
    FS = (10, 8)
    fig, ax = plt.subplots(figsize=FS)
    # Make points translucent so we can visually identify regions with a high density of overlapping points
    ax.scatter(df_item.x, df_item.y, alpha=.1)
    ax.set_title('Movie Clusters')
    fig.savefig(os.path.join(DATA_DIR, 'movie_cluster.png'))

    tsne = TSNE(random_state=1, n_iter=15000, metric="cosine")
    user_embeddings = model.user_tower(torch.tensor(df_user['userId'], dtype=torch.long).to(device),
                                        torch.tensor(df_user['age_group'], dtype=torch.long).to(device),
                                        torch.tensor(df_user['gender'], dtype=torch.long).to(device),
                                        torch.tensor(df_user['profession'], dtype=torch.long).to(device))
    user_embeddings = user_embeddings.detach().cpu().numpy()
    embs = tsne.fit_transform(user_embeddings)
    user_indx, movie_indx = create_vector_index(user_embeddings, movie_embeddings)
    df_user['x'] = embs[:, 0]
    df_user['y'] = embs[:, 1]
    FS = (10, 8)
    fig, ax = plt.subplots(figsize=FS)
    # Make points translucent so we can visually identify regions with a high density of overlapping points
    ax.scatter(df_item.x, df_item.y, alpha=.1)
    ax.set_title('User Clusters')
    fig.savefig(os.path.join(DATA_DIR, 'user_cluster.png'))
    return user_indx, movie_indx

def create_vector_index(user_embeddings, movie_embeddings):
    movie_index = faiss.IndexFlatIP(128)
    movie_index.add(movie_embeddings)
    user_index = faiss.IndexFlatIP(128)
    user_index.add(user_embeddings)
    return user_index, movie_index

def train():

    # User Data
    df_user = load_user_data(os.path.join(DATA_DIR, 'u.user'))
    le_age = LabelEncoder()
    le_age.fit(df_user['age_group'])
    le_job = LabelEncoder()
    le_job.fit(df_user['profession'])
    le_gender = LabelEncoder()
    le_gender.fit(df_user['gender'])
    encodeFeatures(df_user, le_age, le_job, le_gender)

    # Item Data (Movies)
    df_item = load_item(os.path.join(DATA_DIR, 'u.item'))
    
    # User-Item Data 
    df_data = load_data(os.path.join(DATA_DIR, 'u.data'))
    df_data_train = load_data(os.path.join(DATA_DIR, 'ua.base'))
    df_data_test = load_data(os.path.join(DATA_DIR, 'ua.test'))

    num_genres = len(df_item.drop(columns=['itemId', 'title']).iloc[0].values)
    num_age_groups = df_user['age_group'].nunique()+1
    num_professions = df_user['profession'].nunique()+1
    num_users = df_user['userId'].max()+1
    num_movies = df_item['itemId'].nunique()+1

    merged_df = get_mergedDF(df_data, df_user, df_item)
    merged_df_train = get_mergedDF(df_data_train, df_user, df_item)
    merged_df_test = get_mergedDF(df_data_test, df_user, df_item)

    whole_dataset = MovieDatasetWithFeatures(merged_df_train, device)
    train_dataset, val_dataset = torch.utils.data.random_split(whole_dataset, [int(len(whole_dataset)*0.8), len(whole_dataset)-int(len(whole_dataset)*0.8)])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False)


    num_epochs=20
    model = MovieTwoTowerModel(num_users, num_movies, num_age_groups, num_professions, num_genres, 64).to(device)
    model.fit(num_epochs, train_dataloader, val_dataloader)
    model.get_early_stopper().load_best_model(model)
    model.early_stopper.load_best_model(model)
    torch.save(model.state_dict(), os.path.join(DATA_DIR, 'model.pt'))
    plot_training_val_loss(model)
    evaluate(model, merged_df_test)
    user_idx, movie_idx = manifold_plots(model, df_item, df_user)
    return {
        'model': model,
        'user_index': user_idx,
        'movie_index': movie_idx,
        'df_item': df_item,
        'df_data': df_data,
        'df_user': df_user
    }