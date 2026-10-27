"""Данные, признаки и общий протокол оценки top-K."""

from collections import Counter
from pathlib import Path
import ast
import gzip
import hashlib
import html
import re
import shutil
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy import sparse


SEED = 42
METADATA_AVAILABLE_AT = int(pd.Timestamp('2015-09-24T20:33:51Z').timestamp() * 1000)
GENRES = {
    'Pop', 'Rock', 'Classical', 'Jazz', 'Blues', 'Country', 'Folk',
    'R&B', 'Rap & Hip-Hop', 'Dance & Electronic', 'Metal', 'World Music',
    'New Age', 'Reggae', 'Gospel', "Children's Music", 'Soundtracks', 'Latin Music',
    'Christian & Gospel', 'Alternative Rock', 'Indie & Alternative',
}
FILES = {
    'ratings_0core.csv.gz': (
        'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/'
        'benchmark/0core/rating_only/CDs_and_Vinyl.csv.gz',
        '1bade4e428efcffb549b6b7cb1f62a8cde9644e19a981a210b11a4c9c1561e75'),
    'meta_CDs_and_Vinyl_2014.json.gz': (
        'https://snap.stanford.edu/data/amazon/productGraph/'
        'categoryFiles/meta_CDs_and_Vinyl.json.gz',
        '81f7c941bd1fcce57125a277b4974573436fff68be705c47968edb7a6f3be4a2'),
}


def download_data(directory='data'):
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)
    for name, (url, expected) in FILES.items():
        path = directory / name
        if not path.exists():
            print('Загрузка:', name, flush=True)
            temporary = path.with_suffix(path.suffix + '.part')
            with urlopen(url, timeout=120) as source, temporary.open('wb') as dest:
                shutil.copyfileobj(source, dest)
            temporary.replace(path)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != expected:
            raise ValueError(f'SHA256 не совпал: {path}; источник мог измениться')
    return directory


def prepare_data(directory, max_items=5000, min_history=5):
    raw = pd.read_csv(Path(directory) / 'ratings_0core.csv.gz')
    fields = ['user_id', 'parent_asin', 'rating', 'timestamp']
    missing = raw[fields].isna().sum()
    exact_duplicates = int(raw.duplicated(fields).sum())
    clean = raw.dropna(subset=fields).copy()
    clean = clean[clean.rating.between(1, 5) & (clean.timestamp > 0)]
    clean = clean.sort_values(['timestamp', 'user_id', 'parent_asin', 'rating'])
    # Для повторной оценки пары использовано первое наблюдение.
    pair_duplicates = int(clean.duplicated(['user_id', 'parent_asin']).sum())
    clean = clean.drop_duplicates(['user_id', 'parent_asin'], keep='first')
    cutoff1, cutoff2 = [int(clean.timestamp.quantile(q, interpolation='higher'))
                        for q in (0.6, 0.8)]
    assert METADATA_AVAILABLE_AT < cutoff1, 'Архив метаданных должен предшествовать рекомендациям'
    clean['split'] = np.where(clean.timestamp < cutoff1, 'train',
                             np.where(clean.timestamp < cutoff2, 'validation', 'test'))
    train_all = clean[clean.split == 'train']
    counts = train_all.groupby('parent_asin').size().rename('count').reset_index()
    chosen = counts.sort_values(['count', 'parent_asin'], ascending=[False, True])
    selected_items = chosen.parent_asin.head(max_items)
    limited = train_all[train_all.parent_asin.isin(selected_items)]
    activity = limited.groupby('user_id').size()
    user_ids = np.sort(activity[activity >= min_history].index.to_numpy())
    sample = clean[clean.user_id.isin(user_ids)
                   & clean.parent_asin.isin(selected_items)].copy()
    item_ids = np.sort(sample.loc[sample.split == 'train', 'parent_asin'].unique())
    sample = sample[sample.parent_asin.isin(item_ids)].copy()
    sample['user_idx'] = pd.Index(user_ids).get_indexer(sample.user_id)
    sample['item_idx'] = pd.Index(item_ids).get_indexer(sample.parent_asin)
    assert sample[['user_idx', 'item_idx']].min().min() >= 0
    assert 100_000 <= len(sample) <= 2_000_000
    assert len(user_ids) >= 1000 and len(item_ids) >= 1000
    parts = {name: sample[sample.split == name].reset_index(drop=True)
             for name in ['train', 'validation', 'test']}
    assert parts['train'].timestamp.max() < parts['validation'].timestamp.min()
    assert parts['validation'].timestamp.max() < parts['test'].timestamp.min()
    shape = (len(user_ids), len(item_ids))
    report = {
        'raw_rows': len(raw), 'clean_rows': len(clean),
        'raw_users': raw.user_id.nunique(), 'raw_items': raw.parent_asin.nunique(),
        'clean_users': clean.user_id.nunique(), 'clean_items': clean.parent_asin.nunique(),
        'exact_duplicates': exact_duplicates, 'pair_duplicates': pair_duplicates,
        'missing': missing.to_dict(), 'cutoffs': (cutoff1, cutoff2),
        'shape': shape, 'sample_rows': len(sample),
    }
    # Покрытие относительно всех будущих положительных событий до отбора.
    coverage = []
    for name in ['validation', 'test']:
        positive = clean[(clean.split == name) & (clean.rating >= 4)]
        warm = positive[positive.user_id.isin(user_ids)]
        mapped = warm[warm.parent_asin.isin(item_ids)]
        coverage.append({'split': name, 'all_positive_pairs': len(positive),
                         'positive_pairs_selected_users': len(warm),
                         'positive_pairs_in_catalog': len(mapped),
                         'eval_users': mapped.user_id.nunique(),
                         'user_coverage': mapped.user_id.nunique() / positive.user_id.nunique(),
                         'catalog_coverage_selected_users': len(mapped) / max(len(warm), 1)})
    metadata = load_metadata(Path(directory) / 'meta_CDs_and_Vinyl_2014.json.gz', item_ids)
    assert (metadata.title.ne('') | metadata.categories.map(bool)).any()
    return parts, metadata, user_ids, item_ids, report, pd.DataFrame(coverage)


def load_metadata(path, item_ids):
    wanted = set(item_ids)
    records = {}
    asin_pattern = re.compile(r"'asin': '([^']+)'")
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            match = asin_pattern.search(line)
            if match and match[1] in wanted and match[1] not in records:
                obj = ast.literal_eval(line)
                item = match[1]
                title = html.unescape(str(obj.get('title') or '')).strip()
                # Только жанры, без промоакций и ценовых рубрик.
                categories = sorted({c for path in obj.get('categories', [])
                                     for c in path if c in GENRES})
                records[item] = {'parent_asin': item, 'title': title, 'categories': categories,
                                 'in_archive': True}
    return pd.DataFrame([records.get(i, {'parent_asin': i, 'title': '', 'categories': [],
                                       'in_archive': False})
                         for i in item_ids])


def matrix(frame, shape, values=None, positive_only=False):
    if positive_only:
        frame = frame[frame.rating >= 4]
    if values is None:
        values = np.ones(len(frame), dtype=np.float64)
    return sparse.csr_matrix((values, (frame.user_idx, frame.item_idx)), shape=shape)


def item_features(metadata, use_metadata=True, max_words=1000, min_df=5):
    """ID + категории + TF-IDF заголовка; словарь строится по train-каталогу."""
    n_items = len(metadata)
    if not use_metadata:
        return sparse.eye(n_items, format='csr'), {'id': n_items, 'categories': 0, 'words': 0}
    tokenize = lambda text: re.findall(r'[a-z]{2,}', text.lower())
    documents = [Counter(tokenize(title)) for title in metadata.title]
    df = Counter(word for doc in documents for word in doc)
    words = sorted((w for w, c in df.items() if c >= min_df), key=lambda w: (-df[w], w))[:max_words]
    vocab = {w: n_items + j for j, w in enumerate(words)}
    categories = sorted({c for cats in metadata.categories for c in cats})
    cats_vocab = {c: n_items + len(words) + j for j, c in enumerate(categories)}
    rows, cols, values = list(range(n_items)), list(range(n_items)), [1.] * n_items
    for i, (doc, cats) in enumerate(zip(documents, metadata.categories)):
        tokens = [(vocab[w], (1 + np.log(count)) * (np.log((1+n_items)/(1+df[w]))+1))
                  for w, count in doc.items() if w in vocab]
        norm = np.sqrt(sum(v*v for _, v in tokens)) or 1.
        for j, value in tokens:
            rows.append(i); cols.append(j); values.append(value / norm)
        for c in sorted(set(cats)):
            rows.append(i); cols.append(cats_vocab[c]); values.append(1/np.sqrt(len(set(cats))))
    features = sparse.csr_matrix((values, (rows, cols)),
                                shape=(n_items, n_items + len(words) + len(categories)))
    return features, {'id': n_items, 'categories': len(categories), 'words': len(words)}


class Evaluation:
    """Один и тот же каталог, seen-фильтр и релевантность для всех моделей."""

    def __init__(self, history, target, shape, k=10):
        self.seen = matrix(history, shape)
        self.relevance = matrix(target, shape, positive_only=True)
        self.relevance = self.relevance - self.relevance.multiply(self.seen.sign())
        self.relevance.eliminate_zeros()
        self.users = np.flatnonzero(np.diff(self.relevance.indptr))
        self.k = k
        self.shape = shape
        self.discounts = 1 / np.log2(np.arange(2, k + 2))
        assert len(self.users) > 0
        assert (shape[1] - self.seen.getnnz(axis=1)[self.users] >= k).all()

    def recommend(self, model, users):
        scores = np.array(model.score(users), dtype=np.float64, copy=True)
        assert scores.shape == (len(users), self.shape[1])
        assert np.isfinite(scores).all()
        seen = self.seen[users].tocoo()
        scores[seen.row, seen.col] = -np.inf
        # При равенстве score выигрывает меньший item_idx (ASIN по алфавиту).
        top = np.argsort(-scores, axis=1, kind='stable')[:, :self.k]
        assert not np.take_along_axis(self.seen[users].toarray(), top, axis=1).any()
        return top

    def evaluate(self, model, batch_size=256):
        rows = []
        for start in range(0, len(self.users), batch_size):
            users = self.users[start:start + batch_size]
            top = self.recommend(model, users)
            hits = np.take_along_axis(self.relevance[users].toarray(), top, axis=1)
            counts = self.relevance[users].getnnz(axis=1)
            ideal = np.cumsum(self.discounts)[np.minimum(counts, self.k) - 1]
            ndcg = (hits * self.discounts).sum(axis=1) / ideal
            recall = hits.sum(axis=1) / counts
            rows.append(pd.DataFrame({'user_idx': users, 'NDCG@10': ndcg,
                                      'Recall@10': recall, 'relevant_items': counts}))
        details = pd.concat(rows, ignore_index=True)
        metrics = {c: float(details[c].mean()) for c in ['NDCG@10', 'Recall@10']}
        return metrics, details
