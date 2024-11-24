import os
from logging import getLogger
from typing import DefaultDict

from tqdm import tqdm
import numpy as np
import torch
from scipy.sparse import csr_matrix
import torch.nn.functional as F
import random
import pdb
from scipy.sparse import coo_matrix


class PJFPool(object):
    def __init__(self, config):
        self.logger = getLogger()
        self.config = config
        self._load_ids()
        self._load_inter()

    def _load_ids(self):
        for target in ['geek', 'job']:
            token2id = {}
            id2token = []
            filepath = os.path.join(self.config['dataset_path'], f'{target}.token')
            self.logger.info(f'Loading {filepath}')
            with open(filepath, 'r') as file:
                for i, line in enumerate(file):
                    token = line.strip()
                    token2id[token] = i
                    id2token.append(token)
            setattr(self, f'{target}_token2id', token2id)
            setattr(self, f'{target}_id2token', id2token)
            setattr(self, f'{target}_num', len(id2token))

    def _load_inter(self):
        self.geek2jobs = DefaultDict(list)
        self.job2geeks = DefaultDict(list)

        data_all = open(os.path.join(self.config['dataset_path'], f'data.train_all_add'))
        for l in tqdm(data_all):
            gid, jid, label = l.split('\t')
            gid = self.geek_token2id[gid]
            jid = self.job_token2id[jid]
            self.geek2jobs[gid].append(jid)
            self.job2geeks[jid].append(gid)

    def __str__(self):
        return '\n\t'.join(['Pool:'] + [
            f'{self.geek_num} geeks',
            f'{self.job_num} jobs'
        ])

    def __repr__(self):
        return self.__str__()



class DPGNNPool:
    def __init__(self, config):
        self.config = config
        self.logger = getLogger(__name__)

        # Initialize attributes as necessary
        self.geek2jobs = {}
        self.job2geeks = {}
        self.geek_num = 0
        self.job_num = 0

        self.geek_token2id = {}
        self.geek_id2token = []
        self.job_token2id = {}
        self.job_id2token = []

        self.geek_token2bertid = {}
        self.job_token2bertid = {}

        self._load_tokens('geek')
        self._load_tokens('job')

        # Populate geek_token2bertid and job_token2bertid
        self._populate_bert_id_dicts()

        # Load matrices
        self.interaction_matrix = self._load_edge(os.path.join(self.config['dataset_path'], 'data.train_all'))
        self.user_add_matrix = self._load_edge(os.path.join(self.config['dataset_path'], 'data.user_add'))
        self.job_add_matrix = self._load_edge(os.path.join(self.config['dataset_path'], 'data.job_add'))

        self.geek_num = len(self.geek_id2token)
        self.job_num = len(self.job_id2token)

        # Load BERT vectors
        self.u_bert_vec, self.j_bert_vec = self._load_bert_vectors()

    def _load_tokens(self, target):
        token2id = {}
        id2token = []
        filepath = os.path.join(self.config['dataset_path'], f'{target}.token')
        self.logger.info(f'Loading {filepath}')
        with open(filepath, 'r') as file:
            for i, line in enumerate(file):
                token = line.strip()
                token2id[token] = i
                id2token.append(token)
        setattr(self, f'{target}_token2id', token2id)
        setattr(self, f'{target}_id2token', id2token)
        setattr(self, f'{target}_num', len(id2token))

    def _populate_bert_id_dicts(self):
        for token, idx in self.geek_token2id.items():
            self.geek_token2bertid[token] = idx
        for token, idx in self.job_token2id.items():
            self.job_token2bertid[token] = idx

    def _load_edge(self, filepath):
        self.logger.info(f'Loading edges from {filepath}')
        data = []
        row = []
        col = []
        with open(filepath, 'r') as file:
            for line in file:
                user, item, *_ = line.strip().split()
                user = int(user)
                item = int(item)
                row.append(user)
                col.append(item)
                data.append(1)
        return csr_matrix((data, (row, col)))

    def _load_bert_vectors(self):
        u_filepath = os.path.join(self.config['dataset_path'], 'geek.bert.npy')
        j_filepath = os.path.join(self.config['dataset_path'], 'job.bert.npy')
        self.logger.info(f'Loading BERT vectors from {u_filepath} and {j_filepath}')
        u_bert_vec = np.load(u_filepath)
        j_bert_vec = np.load(j_filepath)
        return u_bert_vec, j_bert_vec


class DPGNNPool2(PJFPool):
    def __init__(self, config):
        super(DPGNNPool, self).__init__(config)
        success_file = os.path.join(self.config['dataset_path'], f'data.train_all')
        self.interaction_matrix = self._load_edge(success_file)

        user_add_file = os.path.join(self.config['dataset_path'], f'data.user_add')
        job_add_file = os.path.join(self.config['dataset_path'], f'data.job_add')

        # add_sample_rate = config['add_sample_rate']
        self.user_add_matrix = self._load_edge(user_add_file)
        self.job_add_matrix = self._load_edge(job_add_file)

        if(config['ADD_BERT']):
            self._load_bert()

    def _load_edge(self, filepath):
        self.logger.info(f'Loading from {filepath}')
        self.geek_ids, self.job_ids, self.labels = [], [], []
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                geek_token, job_token, label = line.strip().split('\t')[:3]

                geek_id = self.geek_token2id[geek_token]
                job_id = self.job_token2id[job_token]
                self.geek_ids.append(geek_id)
                self.job_ids.append(job_id)
                self.labels.append(int(label))

        self.geek_ids = torch.LongTensor(self.geek_ids)
        self.job_ids = torch.LongTensor(self.job_ids)
        self.labels = torch.FloatTensor(self.labels)
        
        src = self.geek_ids
        tgt = self.job_ids
        data = self.labels
        interaction_matrix = coo_matrix((data, (src, tgt)), shape=(self.geek_num, self.job_num))
        return interaction_matrix

    def _load_bert(self):
        u_filepath = os.path.join(self.config['dataset_path'], 'geek.bert.npy')
        self.logger.info(f'Loading from {u_filepath}')
        j_filepath = os.path.join(self.config['dataset_path'], 'job.bert.npy')
        # bert_filepath = os.path.join(self.config['dataset_path'], f'data.{self.phase}.bert.npy')
        self.logger.info(f'Loading from {j_filepath}')

        u_array = np.load(u_filepath).astype(np.float64)
        # add padding 
        u_array = np.vstack([u_array, np.zeros((1, u_array.shape[1]))])

        j_array = np.load(j_filepath).astype(np.float64)
        # add padding
        j_array = np.vstack([j_array, np.zeros((1, j_array.shape[1]))])

        self.geek_token2bertid = {}
        self.job_token2bertid = {}
        for i in range(u_array.shape[0]):
            self.geek_token2bertid[str(u_array[i, 0].astype(int))] = i
        for i in range(j_array.shape[0]):
            self.job_token2bertid[str(j_array[i, 0].astype(int))] = i

        self.u_bert_vec = torch.FloatTensor(u_array[:, 1:])
        self.j_bert_vec = torch.FloatTensor(j_array[:, 1:])
