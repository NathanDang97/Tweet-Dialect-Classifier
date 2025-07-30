from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import os

# List of model names and short IDs to use for directory naming
models = {
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "roberta": "cardiffnlp/twitter-roberta-base-sentiment"  # You can use "roberta-base" if preferred
}

# Downloads the TweetEval 'sentiment' dataset and tokenizes it using a Hugging Face tokenizer.
# The processed dataset is saved to disk for reuse.
def download_and_tokenize(save_path_raw: str="../data/tweeteval/raw/",
                          save_path_processed: str="../data/tweeteval/processed/"):
    print("====== BEGIN: Downloading and Tokenizing the Dataset ======")

    # load the TWEETEval dataset
    print("\n- Loading TWEETEval sentiment dataset...\n")
    dataset: DatasetDict = load_dataset("tweet_eval", "sentiment")

    # save the raw dataset
    print(f"\n- Saving to: {save_path_raw}\n")
    os.makedirs(save_path_raw, exist_ok=True)
    dataset.save_to_disk(save_path_raw)

    # Iterate over models
    for short_name, model_name in models.items():
        print(f"\n=== Tokenizing for {short_name.upper()} ===")

        # load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        def tokenize_fn(example):
            return tokenizer(example["text"], padding="max_length", truncation=True)

        print("- Tokenizing dataset...")
        tokenized_dataset = dataset.map(tokenize_fn, batched=True)

        print("- Setting format to PyTorch tensors...")
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        # save the tokenized dataset
        model_dir = os.path.join(save_path_processed, short_name)
        os.makedirs(model_dir, exist_ok=True)
        print(f"- Saving tokenized dataset to: {model_dir}")
        tokenized_dataset.save_to_disk(model_dir)

    print("\n====== FINISH: Downloading and Tokenizing the Dataset ======\n")

if __name__ == "__main__":
    download_and_tokenize()
