import os
import re
import numpy as np
import torch
from transformers import BertTokenizer, BertModel,BertConfig


def clean_str_BERT(string):
    """
    Clean the input string by removing certain patterns and HTML tags.
    """
    patterns = r"\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
    string = string.split('http')[0]
    cleaner = re.compile('<.*?>')
    string = re.sub(cleaner, ' ', string)
    string = re.sub(patterns, ' ', string)
    return string.strip()


def load_model_and_tokenizer(dataset):
    """
    Load BERT model and tokenizer based on the dataset.
    """
    if dataset == 'weibo':
        model_path = '../model/bert-base-chinese/'
    elif dataset == 'pheme':
        model_path = '../model/bert-base-uncased/'
    else:
        raise ValueError("Invalid dataset name")


    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    return tokenizer, model


def load_mappings(node2idx_path, mid2text_path):
    """
    Load node to index and MID to text mappings.
    """
    node2idx_dict = {}
    mid2text_dict = {}

    with open(node2idx_path, 'r', encoding='utf-8') as f:
        for line in f:
            node, idx = line.strip('\n').split('\t')
            node2idx_dict[node] = int(idx)

    with open(mid2text_path, 'r', encoding='utf-8') as f:
        for line in f:
            mid, text = line.strip('\n').split('\t')
            mid2text_dict[mid] = text

    return node2idx_dict, mid2text_dict


def save_embeddings(embeddings, save_path):
    """
    Save BERT embeddings to a file.
    """
    # Detach the tensor from the computation graph and convert to NumPy array
    embeddings_np = embeddings.detach().cpu().numpy()

    # If file already exists, load existing embeddings and append new embeddings
    if os.path.exists(save_path):
        existing_embeddings = np.load(save_path)
        embeddings_np = np.concatenate([existing_embeddings, embeddings_np])
    # Save the combined embeddings to the file
    np.save(save_path, embeddings_np)


def process_texts_and_save_embeddings(texts, tokenizer, model, save_path, batch_size=100):
    """
    Process texts in batches, generate BERT embeddings, and append them to a single file.
    """
    with torch.no_grad():
        start_idx = 0

        while start_idx < len(texts):

            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]

            embeddings = []
            for text in batch_texts:
                cleaned_text = clean_str_BERT(text)
                encoded_input = tokenizer(cleaned_text, return_tensors='pt',truncation=True, max_length=30)
                output = model(**encoded_input)
                pooler_output = output['pooler_output']
                embeddings.append(pooler_output)

            embeddings = torch.cat(embeddings, dim=0)
            # Append embeddings to file
            save_embeddings(embeddings, save_path)

            start_idx = end_idx

            print(f"Processed {end_idx}/{len(texts)}")

    print('Embeddings processing and saving completed.')


def main(dataset, batch_size=100):
    """
    Main function to generate BERT embeddings for the given dataset.
    """
    tokenizer, model = load_model_and_tokenizer(dataset)

    data_dir = os.path.join("../", "data", dataset)
    node2idx_path = os.path.join(data_dir, "node2idx_mid.txt")
    mid2text_path = os.path.join(data_dir, "mid2text.txt")
    save_path = os.path.join(os.path.join(data_dir,"{}_temporal_data".format(dataset)), "text_embeddings.npy")

    node2idx_dict, mid2text_dict = load_mappings(node2idx_path, mid2text_path)

    texts = [mid2text_dict[node] for node in node2idx_dict]

    process_texts_and_save_embeddings(texts, tokenizer, model, save_path, batch_size)


if __name__ == '__main__':
    main('pheme', batch_size=1000)

