import os
import zipfile
import csv
import re
import pandas as pd
import nltk
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# from nltk import pos_tag, word_tokenize
# from nltk.util import ngrams

# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("averaged_perceptron_tagger_eng")

SEED = 42

# helper method to extract .zip file with 
def extract_zip_with_progress_bar(zip_path, save_path_raw, check_file="twitteraae_all"):
    # check if the .zip file is already extracted
    data_dir = os.path.join(save_path_raw, "TwitterAAE-full-v1")
    expected_path = os.path.join(data_dir, check_file)
    if os.path.exists(expected_path):
        print(f"\tThe required data file was already extracted at {expected_path}.")
        return
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = z.namelist()
        for file in tqdm(file_list, desc="Extracting .zip", unit="file"):
            z.extract(member=file, path=save_path_raw)

# helper method to check if the raw data file has already been processed
def check_if_raw_file_processed(save_path_processed):
    expected_path_white = os.path.join(save_path_processed, "white_df.csv")
    expected_path_aave = os.path.join(save_path_processed, "aave_df.csv")
    expected_path_aae_no_aave = os.path.join(save_path_processed, "aae_no_aave_df.csv")
    if os.path.exists(expected_path_white) or os.path.exists(expected_path_aave) or os.path.exists(expected_path_aae_no_aave):
        print("\tThe raw data file has already been processed.")
        return True
    
    return False

# helper method to check if a has file already existed in the disk
def check_if_file_exists(save_path_processed, check_file):
    expected_path = os.path.join(save_path_processed, check_file)
    if os.path.exists(expected_path):
        print(f"\tThe file {check_file} has already existed at {expected_path}.")
        return True
    
    return False

# helper method to check if the data has already been tokenized
def check_if_tokenized(save_path_tokenized):
    expected_file = 'dataset_dict.json'
    expected_path = os.path.join(save_path_tokenized, expected_file)
    if os.path.exists(expected_path):
        print("The data is already tokenized.")
        return True

    return False

# helper method to read tsv file with progress bar
def read_tsv_with_progress_bar(path, chunk_size=100_000):
    total_lines = sum(1 for _ in open(path, "r", encoding="utf-8"))
    chunks = []
    with pd.read_csv(
        path, 
        sep="\t",
        header=None,
        engine="python",
        on_bad_lines="skip",
        chunksize=chunk_size,
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
    ) as reader:
        for chunk in tqdm(reader, total=total_lines // chunk_size, desc="Parsing .tsv"):
            chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)

# helper method to filer the demographic based on the given probabilities
def get_demographic(row):
    probs = {
        "AA": row["prob_aa"],
        "Hispanic": row["prob_hispanic"],
        "Other": row["prob_other"],
        "White": row["prob_white"]
    }
    return max(probs, key=probs.get)

# helper method to filter out tweets written in AAVE
def filter_aave_like_tweets(df, text_column="text"):
    aave_keywords = [
        r"\bfinna\b", r"\bgon\b", r"\biont\b", r"\bain'?t\b", r"\bima\b",
        r"\btryna\b", r"\bwanna\b", r"\bgotta\b", r"\bgimme\b", r"\bgon\b",
        r"\bwoke\b", r"\byall\b", r"\bholla\b", r"\bnah\b", r"\bbruh\b",
        r"\bdat\b", r"\bdis\b", r"\bwassup\b", r"\bbae\b", r"\bdoe\b",
        r"\bchile\b", r"\bngga\b", r"\bnigga\b", r"\bn!gga\b", r"\bbouta\b",
        r"\bbeen\b", r"\bbet\b", r"\bcap\b", r"\bno cap\b", r"\bgonna\b",
        r"\bamp\b", r"\bclapback\b", r"\bfrfr\b", r"\blil\b", r"\bbussin\b"
    ]
    pattern = re.compile("|".join(aave_keywords), re.IGNORECASE)
    return df[df[text_column].str.contains(pattern, na=False)]

# # Note: uncomment the two methods below if you want to use n-grams and POS tags
# # helper method to append pos tags before tokenizing
# def append_pos_tags(example):
#     tokens = word_tokenize(example['text'])
#     tags = pos_tag(tokens)
#     tagged_text = ' '.join(f"{word}_{tag}" for word, tag in tags)
#     example['text'] = tagged_text
#     return example

# # helper method to append n_grams before tokenizing
# def append_ngrams(example, n=2):
#     tokens = word_tokenize(example['text'])
#     bigrams = ['_'.join(bigram) for bigram in ngrams(tokens, n)]
#     example['text'] += ' ' + ' '.join(bigrams)
#     return example

# helper method to flag tweets with zero copula
def flag_zero_copula(example):
    text = example["text"].lower()
    
    # Build regex pattern to catch "[pronoun] [adjective or verb]" without copula
    zero_copula_patterns = [
        # he tall, she smart, they tired
        r"\b(he|she|they|we|you|i)\s+(tall|short|smart|tired|mad|sad|happy|hungry|sleepy|crazy|wild|cold|hot|funny|weird|broke|late)\b",
        # he running, they working
        r"\b(he|she|they|we|you|i)\s+[a-z]+ing\b",
    ]
    
    for pattern in zero_copula_patterns:
        if re.search(pattern, text) and not re.search(r"\b(is|are|am|was|were)\b", text):
            example["text"] += " ZERO_COPULA"
            break

    return example

# helper method to flag habitual be in tweets that is common in slang
def flag_habitual_be(example):
    text = example["text"].lower()

    # Look for subject + "be" + verb-ing (e.g., she be working)
    pattern = r"\b(he|she|they|we|you|i)\s+be\s+[a-z]+ing\b"
    if re.search(pattern, text):
        example["text"] += " HABITUAL_BE"
    
    return example

# helper method to flag double negation in tweets that is common in slang
def flag_double_negation(example):
    text = example["text"].lower()
    
    # If multiple negative tokens appear together, flag it
    negations = ["no", "not", "don't", "didn't", "ain't", "never", "nothing", "nowhere", "none", "nobody"]
    count = sum(text.count(neg) for neg in negations)
    
    if count >= 2:
        example["text"] += " DOUBLE_NEGATION"
    
    return example

# main pipeline
def download_and_prepare(save_path_raw: str="../data/twitteraae/raw",
                         save_path_processed: str="../data/twitteraae/processed",
                         save_path_tokenized: str="../data/twitteraae/processed/tokenized"):
    os.makedirs(save_path_raw, exist_ok=True)

    zip_url = "https://slanglab.cs.umass.edu/TwitterAAE/TwitterAAE-full-v1.zip"
    zip_path = os.path.join(save_path_raw, "TwitterAAE-full-v1.zip")

    print("====== BEGIN: Downloading Blodgett et al. TweeterAAE Dataset ======")
    # download the dataset from the specified url
    if not os.path.exists(zip_path):
        import urllib.request
        urllib.request.urlretrieve(zip_url, zip_path)

    # extract the .zip file
    print("\n- Extracting the data file...")
    extract_zip_with_progress_bar(zip_path, save_path_raw)

    data_dir = os.path.join(save_path_raw, "TwitterAAE-full-v1")
    full_file_path = os.path.join(data_dir, "twitteraae_all")
    
    print("\n- Loading and Processing the raw data...")
    # load and process the raw data file if not already
    if not check_if_raw_file_processed(save_path_processed):
        # parse all examples in the all.tsv file
        print("\tParsing all examples...")
        data = read_tsv_with_progress_bar(full_file_path)
        data.columns = ["id", "timestamp", "userID", "lonlat", "census", "text", 
                        "prob_aa", "prob_hispanic", "prob_other", "prob_white"]

        # process the raw data
        print("\tProcessing the examples...")
        print("\t\tAssigning a demographic to each example...")
        data["demographic"] = data.apply(get_demographic, axis=1) # assign demopraphic based on the probabilities
        data = data.dropna(subset=["demographic"])                # drop the NaN demographic-value if any presents

        # filter data subsets for 3 categories: White, AAVE, and AAE without AAVE
        print("\t\tFiltering data subsets for Standard English, AAVE, and AAE without AAVE...")
        white_df = data[data["demographic"] == "White"].copy()
        aae_all_df = data[data["demographic"] == "AA"].copy()
        aave_df = filter_aave_like_tweets(aae_all_df).copy()
        aae_no_aave_df = aae_all_df[~aae_all_df.index.isin(aave_df.index)].copy()

        # Assign numeric labels
        print("\t\tAssigning numeric labels for each category...")
        white_df["label"] = 0           # Standard English (White)
        aae_no_aave_df["label"] = 1     # AAE minus AAVE
        aave_df["label"] = 2            # AAVE

    print("\n- Preparing the Standard English dataset...")
    # save the Standard English data subset to disk
    os.makedirs(save_path_processed, exist_ok=True)
    white_df_path = os.path.join(save_path_processed, "white_df.csv")
    if not check_if_file_exists(save_path_processed, "white_df.csv"):
        print(f"\tSaving the Standard English data subset with {len(white_df)} rows to: {white_df_path}")
        white_df[["text", "label"]].to_csv(white_df_path, index=False)
    else:
        white_df = pd.read_csv(white_df_path)

    print("\n- Preparing the AAVE dataset...")
    # save the AAVE data subset to disk
    aave_df_path = os.path.join(save_path_processed, "aave_df.csv")
    if not check_if_file_exists(save_path_processed, "aave_df.csv"):
        print(f"\tSaving the AAVE data subset with {len(aave_df)} rows to: {aave_df_path}")
        aave_df[["text", "label"]].to_csv(aave_df_path, index=False)
    else:
        aave_df = pd.read_csv(aave_df_path)

    print("\n- Prepare the AAE (no AAVE) dataset...")
    # save the AAE minus AAVE data subset to disk
    aae_no_aave_df_path = os.path.join(save_path_processed, "aae_no_aave_df.csv")
    if not check_if_file_exists(save_path_processed, "aae_no_aave_df.csv"):
        print(f"\tSaving the AAE (no AAVE) data subset with {len(aae_no_aave_df)} rows to: {aae_no_aave_df_path}")
        aae_no_aave_df[["text", "label"]].to_csv(aae_no_aave_df_path, index=False)
    else:
        aae_no_aave_df = pd.read_csv(aae_no_aave_df_path)

    print("\n- Dataset's Size Summary:")
    print(f"\tStandard English: {len(white_df)} rows")
    print(f"\tAAVE: {len(aave_df)} rows")
    print(f"\tAAE (no AAVE): {len(aae_no_aave_df)} rows")

    # prepare the full dataset
    print("\n- Preparing the full combined dataset...")
    full_df_path = os.path.join(save_path_processed, "full_df.csv")
    if not check_if_file_exists(save_path_processed, "full_df.csv"):

        print("\tBalancing the sub-datasets...")
        target_size = len(aave_df) # the smallest df based on the summary
        print("\t\tDownsampling the AAE (no AAVE) dataset...")
        aae_no_aave_df = aae_no_aave_df.sample(n=target_size, random_state=SEED)
        print("\t\tDownsampling the Standard English dataset...")
        white_df = white_df.sample(n=target_size, random_state=SEED)

        # to debug the downsampling, comment out if needed
        print("\tDataset's Size Summary (after balancing):")
        print(f"\t\tStandard English: {len(white_df)} rows")
        print(f"\t\tAAVE: {len(aave_df)} rows")
        print(f"\t\tAAE (no AAVE): {len(aae_no_aave_df)} rows")

        print("\tConcatenating the sub-datasets...")
        full_df = pd.concat([white_df, aae_no_aave_df, aave_df], ignore_index=True)

        # shuffle the full dataset
        print("\tShuffling the full dataset...")
        full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        # save the full dataset
        print(f"\tSaving the full datset with {len(full_df)} rows to: {full_df_path}")
        full_df[["text", "label"]].to_csv(full_df_path, index=False)
    else:
        full_df = pd.read_csv(full_df_path)

    print(f"\n- The combined dataset contains {len(full_df)} rows.")

    # tokenizing the full dataset
    print("\n- Tokenizing the full dataset...")
    tokenized_data_path = save_path_processed + "/tokenized"
    expected_tokenized_data_path = tokenized_data_path + "/data_dict.json"
    if not check_if_tokenized(expected_tokenized_data_path):
        # convert the format of the dataset
        dataset = Dataset.from_pandas(full_df)

        # train-test split
        temp_split = dataset.train_test_split(test_size=0.2, seed=SEED)
        val_test_split = temp_split['test'].train_test_split(test_size=0.5, seed=SEED)
        final_splits = {
            'train': temp_split['train'],
            'validation': val_test_split['train'],
            'test': val_test_split['test']
        }
        dataset = DatasetDict(final_splits)

        # apply linguistically informed features 
        print("Applying linguistically informed features:")
        # print("POS-tags...")
        # dataset = dataset.map(append_pos_tags)
        # print("N-Grams...")
        # dataset = dataset.map(append_ngrams)
        print("Zero Copula tags...")
        dataset = dataset.map(flag_zero_copula)
        print("Habitual-be tags...")
        dataset = dataset.map(flag_habitual_be)
        print("Double-negation tags...")
        dataset = dataset.map(flag_double_negation)

        # load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
        print("Tokenizing with", tokenizer.name_or_path)
        def tokenize_function(batch):
            return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

        # tokenize the data and set the data format for PyTorch
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        # save the tokenized dataset to disk
        os.makedirs(tokenized_data_path, exist_ok=True)
        print(f"\n- Saving the tokenized dataset to: {tokenized_data_path}")
        dataset.save_to_disk(tokenized_data_path)

    # else:
    #     dataset = load_dataset(tokenized_data_path)


    print("\n====== END: Downloading Blodgett et al. TweeterAAE Dataset ======\n")

if __name__ == "__main__":
    download_and_prepare()