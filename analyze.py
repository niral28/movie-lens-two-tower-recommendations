import matplotlib.pyplot as plt 
import os
import pandas as pd
import torch
from sklearn.manifold import TSNE

from dataset import MovieDatasetWithFeatures, DistilBertTokenizer

DATA_DIR='data'

from fuzzywuzzy import process

def movie_finder(title, df):
  all_titles = df['title'].tolist()
  closest_match = process.extractOne(title, all_titles)
  return closest_match


try:
    from adjustText import adjust_text
except ImportError:
    def adjust_text(*args, **kwargs):
        pass
    
def manifold_plots(model: MovieDatasetWithFeatures, df_item: pd.DataFrame, df_user:pd.DataFrame):
    def adjust_text(*args, **kwargs):
        pass

    def plot_bg(bg_alpha=.01, figsize=(13, 9), emb_2d=None):
        """Create and return a plot of all our movie embeddings with very low opacity.
        (Intended to be used as a basis for further - more prominent - plotting of a
        subset of movies. Having the overall shape of the map space in the background is
        useful for context.)
        """
        if emb_2d is None:
            emb_2d = embs
        fig, ax = plt.subplots(figsize=figsize)
        X = emb_2d[:, 0]
        Y = emb_2d[:, 1]
        ax.scatter(X, Y, alpha=bg_alpha)
        return ax

    def annotate_sample(df, n, n_ratings_thresh=0):
        """Plot our embeddings with a random sample of n movies annotated.
        Only selects movies where the number of ratings is at least n_ratings_thresh.
        """
        sample = df[df.n_ratings >= n_ratings_thresh].sample(
            n, random_state=1)
        plot_with_annotations(df, sample.index)

    def plot_by_title_pattern(df, pattern, **kwargs):
        """Plot all movies whose titles match the given regex pattern.
        """
        match = df[df.title.str.contains(pattern)]
        return plot_with_annotations(df, match.index, **kwargs)

    def add_annotations(ax, label_indices, emb_2d=None, **kwargs):
        if emb_2d is None:
            emb_2d = embs
        X = emb_2d[label_indices, 0]
        Y = emb_2d[label_indices, 1]
        ax.scatter(X, Y, **kwargs)

    def plot_with_annotations(df, label_indices, text=True, labels=None, alpha=1, **kwargs):
        ax = plot_bg(**kwargs)
        Xlabeled = embs[label_indices, 0]
        Ylabeled = embs[label_indices, 1]
        if labels is not None:
            for x, y, label in zip(Xlabeled, Ylabeled, labels):
                ax.scatter(x, y, alpha=alpha, label=label, marker='1',
                        s=90,
                        )
            fig.legend()
        else:
            ax.scatter(Xlabeled, Ylabeled, alpha=alpha, color='green')

        if text:
            # TODO: Add abbreviated title column
            titles = df.loc[label_indices, 'title'].values
            texts = []
            for label, x, y in zip(titles, Xlabeled, Ylabeled):
                t = ax.annotate(label, xy=(x, y))
                texts.append(t)
            adjust_text(texts,
                        #expand_text=(1.01, 1.05),
                        arrowprops=dict(arrowstyle='->', color='red'),
                    )
        return ax

    FS = (13, 9)
    def plot_region(df, x0, x1, y0, y1, text=True):
        """Plot the region of the mapping space bounded by the given x and y limits.
        """
        fig, ax = plt.subplots(figsize=FS)
        pts = df[
            (df.x >= x0) & (df.x <= x1)
            & (df.y >= y0) & (df.y <= y1)
        ]
        ax.scatter(pts.x, pts.y, alpha=.6)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        if text:
            texts = []
            for label, x, y in zip(pts.title.values, pts.x.values, pts.y.values):
                t = ax.annotate(label, xy=(x, y))
                texts.append(t)
            adjust_text(texts, expand_text=(1.01, 1.05))
            fig.savefig(os.path.join(DATA_DIR, '{text}_cluster.png'))
        return ax

    def plot_region_around(df, title, margin=5, **kwargs):
        """Plot the region of the mapping space in the neighbourhood of the the movie with
        the given title. The margin parameter controls the size of the neighbourhood around
        the movie.
        """
        xmargin = ymargin = margin
        match = df[df.title == title]
        assert len(match) == 1
        print(match)
        row = match.iloc[0]
        return plot_region(df, row.x-xmargin, row.x+xmargin, row.y-ymargin, row.y+ymargin, **kwargs)
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
    
    embs = tsne.fit_transform(movie_embeddings.detach().cpu().numpy())
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
    embs = tsne.fit_transform(user_embeddings.detach().cpu().numpy())
    df_user['x'] = embs[:, 0]
    df_user['y'] = embs[:, 1]
    FS = (10, 8)
    fig, ax = plt.subplots(figsize=FS)
    # Make points translucent so we can visually identify regions with a high density of overlapping points
    ax.scatter(df_item.x, df_item.y, alpha=.1)
    ax.set_title('User Clusters')
    fig.savefig(os.path.join(DATA_DIR, 'user_cluster.png'))

    title = movie_finder('James Bond', df_item)[0]
    plot_region_around(df_item, title, 4)

    docs = df_item[ (df_item.comedy == 1) ]
    plot_with_annotations(df_item, docs.index, text=False, alpha=.4, figsize=(15, 8));
    plt.title('Comedy Clusters')
    plt.savefig(os.path.join(DATA_DIR,'comedy_clusters.png'))

    docs = df_item[ (df_item.children == 1) ]
    plot_with_annotations(df_item, docs.index, text=False, alpha=.4, figsize=(15, 8));
    plt.title('Children Clusters')
    plt.savefig(os.path.join(DATA_DIR,'children_clusters.png'))
    
    docs = df_item[ (df_item.romance == 1) ]
    plot_with_annotations(df_item, docs.index, text=False, alpha=.4, figsize=(15, 8));
    plt.title('Romance Clusters')
    plt.savefig(os.path.join(DATA_DIR,'romance_clusters.png'))
    
    docs = df_item[ (df_item.sciFi == 1) ]
    plot_with_annotations(df_item, docs.index, text=False, alpha=.4, figsize=(15, 8));
    plt.title('SciFi Clusters')
    plt.savefig(os.path.join(DATA_DIR,'scifi_clusters.png'))
    
    docs = df_user[ (df_user.age_group == 2) ]
    plot_with_annotations(df_user, docs.index, text=False, alpha=.4, figsize=(15, 8));
    plt.title('Age Group 20-25')
    plt.savefig('Age Group 20-25')