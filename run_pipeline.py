import warnings
warnings.simplefilter('ignore')
from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    train_pipeline("data/olist_customers_dataset.csv")