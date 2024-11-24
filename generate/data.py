import os
import random


def generate_data_file(dataset_path, file_name, num_entries=1000, geek_tokens=None, job_tokens=None, labels=None):
    """
    Generates a sample TSV file with columns geek_token, job_token, and label.
    Args:
        dataset_path (str): The directory where the file should be saved.
        file_name (str): The name of the file to create.
        num_entries (int): The number of entries to generate in the file.
        geek_tokens (list): List of geek tokens.
        job_tokens (list): List of job tokens.
        labels (list): List of labels.
    """
    if geek_tokens is None:
        geek_tokens = [f'geek_{i}' for i in range(0, 100)]
        geek_tokens = [f'{i}' for i in range(0, 100)]
    if job_tokens is None:
        job_tokens = [f'job_{i}' for i in range(0, 100)]
        job_tokens = [f'{i}' for i in range(0, 100)]
    if labels is None:
        labels = [0, 1]

    os.makedirs(dataset_path, exist_ok=True)
    file_path = os.path.join(dataset_path, file_name)

    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            for _ in range(num_entries):
                geek_token = random.choice(geek_tokens)
                job_token = random.choice(job_tokens)
                label = random.choice(labels)
                f.write(f'{geek_token}\t{job_token}\t{label}\n')
        print(f'Sample data file created at {file_path}')
    else:
        print(f'Sample data file is exist at {file_path}')


def generate_token_file(dataset_path, file_name, tokens):
    """
    Generates a token file where each line contains a single token.

    Args:
        dataset_path (str): The directory where the file should be saved.
        file_name (str): The name of the file to create.
        tokens (list): List of tokens to write to the file.
    """
    file_path = os.path.join(dataset_path, file_name)
    with open(file_path, 'w') as f:
        for token in tokens:
            f.write(f'{token}\n')
    print(f'Token file created at {file_path}')


def generate_geek_and_job_tokens(dataset_path, num_geeks=100, num_jobs=100):
    """
    Generates the geek.token and job.token files.

    Args:
        dataset_path (str): The directory where the files should be saved.
        num_geeks (int): The number of geek tokens to generate.
        num_jobs (int): The number of job tokens to generate.
    """
    geek_tokens = [f'geek_{i}' for i in range(0, num_geeks)]
    job_tokens = [f'job_{i}' for i in range(0, num_jobs)]
    geek_tokens = [f'{i}' for i in range(0, num_geeks)]
    job_tokens = [f'{i}' for i in range(0, num_jobs)]
    generate_token_file(dataset_path, 'geek.token', geek_tokens)
    generate_token_file(dataset_path, 'job.token', job_tokens)


def generate_sample_data(dataset_path):
    # Generate data for various types
    data_types = ['train_all_add', 'train_all', 'valid_g', 'valid_j', 'test_g', 'test_j', 'user_add', 'job_add']
    num_entries_dict = {
        'train_all_add': 1000,
        'train_all': 1000,
        'valid_g': 200,
        'valid_j': 200,
        'test_g': 200,
        'test_j': 200,
        'user_add': 100,
        'job_add': 100
    }

    for data_type in data_types:
        generate_data_file(dataset_path, f'data.{data_type}', num_entries=num_entries_dict[data_type])

    # Handle the geek weak file not found case
    geek_weak_path = os.path.join(dataset_path, 'geek.weak')
    if not os.path.exists(geek_weak_path):
        with open(geek_weak_path, 'w') as f:
            f.write(f'{"geek_1"}\t{"weak"}')
        print(f'Placeholder geek.weak file created at {geek_weak_path}')

    generate_geek_and_job_tokens(dataset_path)
