from collections import defaultdict
from dataclasses import dataclass
import functools
import itertools
import json
from multiprocessing import Pool
import os
import random
import re
import gzip
import shutil
import subprocess
import pandas as pd
from typing import Callable, Dict, List, Generator, Tuple
from pandas.io.json._json import JsonReader
from transformers import BertTokenizer

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook as tqdm

import torch
from torch import nn, optim
import torch.cuda.amp 
from pathlib import Path
from torch.cuda.amp import GradScaler as scaler
from torch.utils.data import Dataset, Subset, DataLoader

#@title Example Data
"""
A class of one of the subsets of how an example will be broken down
"""

@dataclass
class Example(object):
    example_id: int
    candidates: List[Dict]
    annotations: Dict
    doc_start: int
    question_len: int
    tokenized_to_original_index: List[int]
    input_ids: List[int]
    start_position: int
    end_position: int
    class_label: str

"""
Similar to the collate_fn but intended for validation
"""
def eval_collate_fn(examples: List[Example]) -> Tuple[List[torch.Tensor], List[Example]]:
    # input tokens
    max_len = max([len(example.input_ids) for example in examples])
    tokens = np.zeros((len(examples), max_len), dtype=np.int64)
    token_type_ids = np.ones((len(examples), max_len), dtype=np.int64)
    for i, example in enumerate(examples):
        row = example.input_ids
        tokens[i, :len(row)] = row
        token_type_id = [0 if i <= row.index(102) else 1
                         for i in range(len(row))]  # 102 corresponds to [SEP]
        token_type_ids[i, :len(row)] = token_type_id
    attention_mask = tokens > 0
    inputs = [torch.from_numpy(tokens),
              torch.from_numpy(attention_mask),
              torch.from_numpy(token_type_ids)]

    return inputs, examples

"""
Takes each validation example and passes through the model for 
future evalution; no training
"""
def eval_model(
    model: nn.Module,
    valid_loader: DataLoader,
    device: torch.device = torch.device('cuda')
) -> Dict[str, float]:
    """Compute validation score.
    
    Parameters
    ----------
    model : nn.Module
        Model for prediction.
    valid_loader : DataLoader
        Data loader of validation data.
    device : torch.device, optional
        Device for computation.
    
    Returns
    -------
    dict
        Scores of validation data.
        `long_score`: score of long answers
        `short_score`: score of short answers
        `overall_score`: score of the competition metric
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        result = Result()
        for inputs, examples in tqdm(valid_loader):
            input_ids, attention_mask, token_type_ids = inputs
            y_preds = model(input_ids.to(device),
                            attention_mask.to(device),
                            token_type_ids.to(device))
            
            start_preds, end_preds, class_preds = (p.detach().cpu() for p in y_preds)
            start_logits, start_index = torch.max(start_preds, dim=1)
            end_logits, end_index = torch.max(end_preds, dim=1)

            # span logits minus the cls logits seems to be close to the best
            cls_logits = start_preds[:, 0] + end_preds[:, 0]  # '[CLS]' logits
            logits = start_logits + end_logits - cls_logits  # (batch_size,)
            indices = torch.stack((start_index, end_index)).transpose(0, 1)  # (batch_size, 2)
            result.update(examples, logits.numpy(), indices.numpy(), class_preds.numpy())

    return result.score()


"""
This class is called in eval_model and each functions is given an overall
description.
"""
class Result(object):
    """Stores results of all test data.
    """
    
    def __init__(self):
        self.examples = {}
        self.results = {}
        self.best_scores = defaultdict(float)
        self.class_labels = ['LONG', 'NO', 'SHORT', 'UNKNOWN', 'YES']
        
    @staticmethod
    def is_valid_index(example: Example, index: List[int]) -> bool:
        """Return whether valid index or not.
        """
        start_index, end_index = index
        if start_index > end_index:
            return False
        if start_index <= example.question_len + 2:
            return False
        return True
        
    def update(
        self,
        examples: List[Example],
        logits: torch.Tensor,
        indices: torch.Tensor,
        class_preds: torch.Tensor
    ):
        """Update batch objects.
        
        Parameters
        ----------
        examples : list of Example
        logits : np.ndarray with shape (batch_size,)
            Scores of each examples..
        indices : np.ndarray with shape (batch_size, 2)
            `start_index` and `end_index` pairs of each examples.
        class_preds : np.ndarray with shape (batch_size, num_classes)
            Class predicition scores of each examples.
        """
        for i, example in enumerate(examples):
            if self.is_valid_index(example, indices[i]) and \
               self.best_scores[example.example_id] < logits[i]:
                self.best_scores[example.example_id] = logits[i]
                self.examples[example.example_id] = example
                self.results[example.example_id] = [
                    example.doc_start, indices[i], class_preds[i]]

    def _generate_predictions(self) -> Generator[Dict, None, None]:
        """Generate predictions of each examples.
        """
        for example_id in self.results.keys():
            doc_start, index, class_pred = self.results[example_id]
            example = self.examples[example_id]
            tokenized_to_original_index = example.tokenized_to_original_index
            short_start_index = tokenized_to_original_index[doc_start + index[0]]
            short_end_index = tokenized_to_original_index[doc_start + index[1]]
            long_start_index = -1
            long_end_index = -1
            for candidate in example.candidates:
                if candidate['start_token'] <= short_start_index and \
                   short_end_index <= candidate['end_token']:
                    long_start_index = candidate['start_token']
                    long_end_index = candidate['end_token']
                    break
            yield {
                'example': example,
                'long_answer': [long_start_index, long_end_index],
                'short_answer': [short_start_index, short_end_index],
                'yes_no_answer': class_pred
            }

    def end(self) -> Dict[str, Dict]:
        """Get predictions in submission format.
        """
        preds = {}
        for pred in self._generate_predictions():
            example = pred['example']
            long_start_index, long_end_index = pred['long_answer']
            short_start_index, short_end_index = pred['short_answer']
            class_pred = pred['yes_no_answer']

            long_answer = f'{long_start_index}:{long_end_index}' if long_start_index != -1 else np.nan
            short_answer = f'{short_start_index}:{short_end_index}'
            class_pred = self.class_labels[class_pred.argmax()]
            short_answer += ' ' + class_pred if class_pred in ['YES', 'NO'] else ''
            preds[f'{example.example_id}_long'] = long_answer
            preds[f'{example.example_id}_short'] = short_answer
        return preds

    def score(self) -> Dict[str, float]:
        """Calculate score of all examples.
        """

        def _safe_divide(x: int, y: int) -> float:
            """Compute x / y, but return 0 if y is zero.
            """
            if y == 0:
                return 0.
            else:
                return x / y

        def _compute_f1(answer_stats: List[List[bool]]) -> float:
            """Computes F1, precision, recall for a list of answer scores.
            """
            has_answer, has_pred, is_correct = list(zip(*answer_stats))
            precision = _safe_divide(sum(is_correct), sum(has_pred))
            recall = _safe_divide(sum(is_correct), sum(has_answer))
            f1 = _safe_divide(2 * precision * recall, precision + recall)
            return f1, precision, recall

        long_scores = []
        short_scores = []
        for pred in self._generate_predictions():
            example = pred['example']
            long_pred = pred['long_answer']
            short_pred = pred['short_answer']
            class_pred = pred['yes_no_answer']
            yes_no_label = self.class_labels[class_pred.argmax()]

            # long score
            long_label = example.annotations['long_answer']
            has_answer = long_label['candidate_index'] != -1
            has_pred = long_pred[0] != -1 and long_pred[1] != -1
            is_correct = False
            if long_label['start_token'] == long_pred[0] and \
               long_label['end_token'] == long_pred[1]:
                is_correct = True
            long_scores.append([has_answer, has_pred, is_correct])

            # short score
            short_labels = example.annotations['short_answers']
            class_pred = example.annotations['yes_no_answer']
            has_answer = yes_no_label != 'NONE' or len(short_labels) != 0
            has_pred = class_pred != 'NONE' or (short_pred[0] != -1 and short_pred[1] != -1)
            is_correct = False
            if class_pred in ['YES', 'NO']:
                is_correct = yes_no_label == class_pred
            else:
                for short_label in short_labels:
                    if short_label['start_token'] == short_pred[0] and \
                       short_label['end_token'] == short_pred[1]:
                        is_correct = True
                        break
            short_scores.append([has_answer, has_pred, is_correct])

        long_score = _compute_f1(long_scores)
        short_score = _compute_f1(short_scores)
        return {
            'long_score': long_score,
            'short_score': short_score,
            'overall_score': (long_score[0] + short_score[0]) / 2
        }