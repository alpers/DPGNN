import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel


def generate_bert_embeddings(tokens, model_name='bert-base-uncased'):
    """
    Generates BERT embeddings for a list of tokens.

    Args:
        tokens (list): List of tokens to generate embeddings for.
        model_name (str): Pre-trained BERT model from Transformers library.

    Returns:
        numpy.ndarray: Array of embeddings.
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    embeddings = []
    with torch.no_grad():
        for token in tokens:
            inputs = tokenizer(token, return_tensors='pt', padding=True, truncation=True)
            outputs = model(**inputs)
            # Take the embedding corresponding to the [CLS] token
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)

    return np.array(embeddings)


def save_npy_file(data, dataset_path, file_name):
    """
    Saves data as a .npy file.

    Args:
        data (numpy.ndarray): Data to be saved.
        dataset_path (str): Path to save the file.
        file_name (str): Name of the .npy file.
    """
    file_path = os.path.join(dataset_path, file_name)
    np.save(file_path, data)
    print(f'{file_name} saved at {file_path}')


def generate_and_save_bert_embeddings(dataset_path, token_file_name, npy_file_name):
    """
    Generates and saves BERT embeddings from token file.

    Args:
        dataset_path (str): Path to dataset.
        token_file_name (str): File name of the token file.
        npy_file_name (str): File name for the .npy file.
    """
    token_file_path = os.path.join(dataset_path, token_file_name)

    with open(token_file_path, 'r') as file:
        tokens = [line.strip() for line in file]

    embeddings = generate_bert_embeddings(tokens)
    save_npy_file(embeddings, dataset_path, npy_file_name)


def generate_sample_model(dataset_path):
    generate_and_save_bert_embeddings(dataset_path, 'geek.token', 'geek.bert.npy')
    generate_and_save_bert_embeddings(dataset_path, 'job.token', 'job.bert.npy')
