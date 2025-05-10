from src.dataset.reddits import RedditSDataset
from src.dataset.movies import MoviesDataset
from src.dataset.toys import ToysDataset
from src.dataset.grocery import GroceryDataset


load_dataset = {
    'reddits': RedditSDataset,
    'movies': MoviesDataset,
    'toys': ToysDataset,
    'grocery': GroceryDataset,
}