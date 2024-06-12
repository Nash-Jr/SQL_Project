import json
import tempfile
import pandas as pd
from sklearn.model_selection import train_test_split
import cohere
import os
import time
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_combine_json_files(spider_path, others_path, output_path):
    try:
        with open(spider_path, 'r') as f:
            train_spider = json.load(f)

        with open(others_path, 'r') as f:
            train_others = json.load(f)

        combined_train_data = train_spider + train_others

        with open(output_path, 'w') as f:
            json.dump(combined_train_data, f)

        df = pd.DataFrame(combined_train_data)
        return df

    except Exception as e:
        logging.error(f"Error loading and combining JSON files: {e}")
        return None


def clean_text(text):
    import re
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text


def clean_dataframe(df):
    df['question'] = df['question'].apply(clean_text)
    df['query'] = df['query'].apply(clean_text)
    return df


def split_data(df):
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_examples = list(zip(train_df['question'], train_df['query']))
    val_examples = list(zip(val_df['question'], val_df['query']))
    return train_examples, val_examples


def prepare_data_for_cohere(train_examples, val_examples):
    train_data = [{"input": q, "output": s} for q, s in train_examples]
    val_data = [{"input": q, "output": s} for q, s in val_examples]
    return train_data, val_data


def train_model(train_data, val_data, cohere_client):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".jsonl") as train_file:
                for entry in train_data:
                    train_file.write(json.dumps(entry) + "\n")
                train_file_path = train_file.name

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".jsonl") as val_file:
                for entry in val_data:
                    val_file.write(json.dumps(entry) + "\n")
                val_file_path = val_file.name

            train_response = cohere_client.datasets.create(
                name="train_dataset",
                data=open(train_file_path, "rb"),
                type="prompt-completion-finetune-input"
            )
            train_dataset_id = train_response.id

            val_response = cohere_client.datasets.create(
                name="val_dataset",
                data=open(val_file_path, "rb"),
                type="prompt-completion-finetune-input"
            )
            val_dataset_id = val_response.id

            cohere_client.wait(train_response)
            cohere_client.wait(val_response)

            fine_tune_response = cohere_client.finetune(
                model_id='command-xlarge-nightly',
                train_dataset_id=train_dataset_id,
                val_dataset_id=val_dataset_id,
                model_name='sql-text-to-sql-model',
                epochs=5,
                batch_size=16,
                learning_rate=2e-5
            )

            logging.info("Fine-tuning job submitted. Response:")
            logging.info(fine_tune_response)
            break
        except Exception as e:
            logging.error(
                f"Error training model (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
            if attempt == max_retries - 1:
                raise e  # Re-raise the last exception if all retries fail


def main():
    spider_path = "c:/Users/nacho/New folder/SQL_Project/spider/spider/train_spider.json"
    others_path = "c:/Users/nacho/New folder/SQL_Project/spider/spider/train_others.json"
    output_path = "c:/Users/nacho/New folder/SQL_Project/combined_train_data.json"

    df = load_and_combine_json_files(spider_path, others_path, output_path)

    if df is not None:
        logging.info(df.head())

        df = clean_dataframe(df)
        train_examples, val_examples = split_data(df)
        train_data, val_data = prepare_data_for_cohere(
            train_examples, val_examples)

        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is not set.")

        cohere_client = cohere.Client(api_key)

        try:
            train_model(train_data, val_data, cohere_client)
        except Exception as e:
            logging.error(f"Failed to train model: {e}")

        logging.info(f"Training examples: {len(train_examples)}")
        logging.info(f"Validation examples: {len(val_examples)}")


if __name__ == "__main__":
    main()
