import json
import math
import re
import string
from collections import defaultdict
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse


# extract words and corresponding labels in the generation texts
def extract(preds_text):
    pattern = r'\(B-.*?\)|\(I-.*?\)|\(O\)'
    words, labels, pre_bound = [], [], 0
    for label_span in re.finditer(pattern, preds_text):
        l, r = label_span.span()
        word, label = preds_text[pre_bound: l], preds_text[l + 1: r - 1]
        if word.strip() != '':
            words.append(word.strip())
            labels.append(label.strip())
        pre_bound = r
    return words, labels

# judge if b exist as a subsequence of a
# if true, return the corresponding match index between a and b
# else return false
def contains_in_order(a, b):
    n, m = len(a), len(b)
    match_idx = [-1] * len(a)
    idx_b = 0
    if m == 0:
        return match_idx
    for i in range(n):
        if a[i] == b[idx_b]:
            match_idx[i] = idx_b
            idx_b += 1
            if idx_b == m:
                return match_idx
    return False

# Traditional LCS solution
# the complexity is O(N^2)
def lcs_solve(a, b):
    n, m = len(a), len(b)
    lcs_arr = [[0] * (m + 1) for _ in range(n + 1)]
    pre_arr = [[(-1, -1) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                lcs_arr[i][j] = lcs_arr[i - 1][j - 1] + 1
                pre_arr[i][j] = (i - 1, j - 1)
            elif lcs_arr[i - 1][j] > lcs_arr[i][j - 1]:
                lcs_arr[i][j] = lcs_arr[i - 1][j]
                pre_arr[i][j] = (i - 1, j)
            else:
                lcs_arr[i][j] = lcs_arr[i][j - 1]
                pre_arr[i][j] = (i, j - 1)
    i, j, match_idx = n, m, [-1] * n
    while i > 0 and j > 0:
        u, v = pre_arr[i][j]
        if i - u == 1 and j - v == 1:
            match_idx[i - 1] = j - 1
        i, j = u, v
    return match_idx

# A fast version of LCS with a complexity of O(NlogN)
# in the condiction that there are few depulicate words in the sentence
# input: a = [word_1, word_2, ..., word_n], b = [word_1, word_2, ..., word_m]
# return: match_idx = [idx_1, idx_2, ..., idx_n] (correspoding matching index between a and b)
def lcs_solve_fast(a, b):
    n, m = len(a), len(b)
    match_idx = [-1] * n
    match_list_b = defaultdict(list)
    
    # First we can convert the LCS problem into a LIS problem,
    # i.e., LCS(a, b) <=> LIS(index_list)
    for idx, word in enumerate(reversed(b)):
        match_list_b[word].append(m - idx - 1)
    index_list = []
    elem_list = []
    for idx, word in enumerate(a):
        if word in match_list_b:
            index_list.extend(match_list_b[word])
            elem_list.extend([idx] * len(match_list_b[word]))

    # then we compute the longest increasing subsequence of index_list
    # we compute a dag, the edges array store the parent of the node, and path store the results
    father, increasing_seq = [[(-1, -1, -1)]], [-1]
    for i in range(len(index_list)):
        if index_list[i] > increasing_seq[-1]:
            father.append([(len(father[-1]) - 1, i, index_list[i])])
            increasing_seq.append(index_list[i])
        else:
            l, r, query_idx = 0, len(increasing_seq) - 1, -1
            while l <= r:
                mid = (l + r) >> 1
                if increasing_seq[mid] >= index_list[i]:
                    query_idx = mid
                    r = mid - 1
                else:
                    l = mid + 1
            father[query_idx].append((len(father[query_idx - 1]) - 1, i, index_list[i]))
            increasing_seq[query_idx] = index_list[i]

    # finally, we trace back the path to get a solution of the original LCS problem
    i, j = len(father) - 1, len(father[-1]) - 1
    while i > 0:
        match_idx[elem_list[father[i][j][1]]] = father[i][j][2]
        j = father[i][j][0]
        i -= 1
    return match_idx


def hierarchical_matching(raw_words, words, labels, tokenizer=None):
    raw_words = list(map(str.lower, raw_words))
    words = list(map(str.lower, words))

    # back tokenization to get a better matching condition
    if tokenizer is not None:
        raw_words = tokenizer.batch_decode(tokenizer(raw_words)["input_ids"], skip_special_tokens=True)

    # Condition 1, raw_words = words
    if raw_words == words:
        return labels

    # Condition 2, words exist as a subsequence of raw_words
    match_idx = contains_in_order(raw_words, words)
    if match_idx is not False:
        match_labels = [labels[idx] if idx != -1 else 'O' for idx in match_idx]
        return match_labels

    # Condition 3, compute LCS(raw_words, words) in O(NlogN)
    match_idx = lcs_solve_fast(raw_words, words)
    match_labels = [labels[idx] if idx != -1 and labels[idx] else 'O' for idx in match_idx]
    return match_labels

# convert the unstructured texts into structured entities
def extract_predictions(example, tokenizer=None):
    pred_words, pred_labels = extract(example['prediction'].strip())
    valid_labels = []
    for label in example['label_list']:
        valid_labels.extend([f'B-{label}', f'I-{label}'])
    for i, label in enumerate(pred_labels):
        if label not in valid_labels:
            pred_labels[i] = "O"
    predictions = hierarchical_matching(example['instance']['words'], pred_words, pred_labels, tokenizer=tokenizer)
    assert len(predictions) == len(example['instance']['labels'])
    return predictions

# normalize answer, 
# cp from https://github.com/universal-ner/universal-ner/blob/main/src/eval/evaluate.py
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

# parser BIO format into entity format, 
# modified from https://github.com/universal-ner/universal-ner/blob/main/src/eval/evaluate.py
def parser(words, labels):
    assert len(words) == len(labels)
    spans_list = []
    span_words, span_label = [], None
    for word, label in zip(words, labels):
        if len(span_words) > 0 and (label[0] == 'B' or label[0] == 'O'):
            spans_list.append((' '.join(span_words), span_label))
            span_words, span_label = [], None
        if label != 'O':
            span_words.append(word)
            span_label = label[2:]
    if span_label is not None:
        spans_list.append((' '.join(span_words), span_label))
    formatted_items = []
    for item in spans_list:
        if isinstance(item, list) or isinstance(item, tuple):
            item = tuple([normalize_answer(element) for element in item])
        else:
            item = normalize_answer(item)
        if item not in formatted_items:
            formatted_items.append(item)
    return formatted_items

# compute F1 score
# modified from https://github.com/universal-ner/universal-ner/blob/main/src/eval/evaluate.py
class NEREvaluator:
    def evaluate(self, examples: list, tokenizer):
        n_correct, n_pos_gold, n_pos_pred = 0, 0, 0
        for example in tqdm(examples):
            words = example['instance']['words']
            labels = example['instance']['labels']
            predictions = extract_predictions(example, tokenizer)
            gold_tuples = parser(words, labels)
            pred_tuples = parser(words, predictions)
            for t in pred_tuples:
                if t in gold_tuples:
                    n_correct += 1
                n_pos_pred += 1
            n_pos_gold += len(gold_tuples)
        prec = n_correct / (n_pos_pred + 1e-10)
        recall = n_correct / (n_pos_gold + 1e-10)
        f1 = 2 * prec * recall / (prec + recall + 1e-10)
        return {
            'precision': prec,
            'recall': recall,
            'f1': f1,
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", default="google/flan-t5-base", type=str, required=True)
    parser.add_argument("--prediction-path", default="model_predictions/flan-t5-xxl-stage1-beam1.jsonl", type=str, required=True)
    args = parser.parse_args()

    # load tokenizer and prediction data
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    all_examples = defaultdict(list)
    with open(args.prediction_path, 'r') as fh:
        for line in fh.readlines():
            line_data = json.loads(line)
            all_examples[line_data['dataset']].append(line_data)

    # evaluate
    tot_f1, tot_dataset = 0, 0
    for dataset in all_examples:
        eval_result = NEREvaluator().evaluate(all_examples[dataset], tokenizer=tokenizer)
        print(f'Dataset: {dataset}, F1: {eval_result["f1"]}, Precision: {eval_result["precision"]}, Recall: {eval_result["recall"]}')
        tot_f1 += eval_result["f1"]
        tot_dataset += 1
    print(f'avg_f1: {tot_f1 / tot_dataset}')


if __name__ == "__main__":
    main()

'''
Example of predictions:
{
    "task": "NER", 
    "dataset": "WikiNeural", 
    "split": "test", 
    "label_list": ["location", "person", "organization"], 
    "negative_boundary": null, 
    "instance": {
        "id": "11596", 
        "subpart": "1", 
        "words": ["This", "system", "was", "widely", "copied", "in", "various", "NATO", "forces", "."], 
        "labels": ["O", "O", "O", "O", "O", "O", "O", "B-organization", "O", "O"], 
        "instruction_inputs": "Please analyze the sentence provided, identifying the type of entity for each word on a token-by-token basis.\nOutput format is: word_1(label_1), word_2(label_2), ...\nWe'll use the BIO-format to label the entities, where:\n1. B- (Begin) indicates the start of a named entity.\n2. I- (Inside) is used for words within a named entity but are not the first word.\n3. O (Outside) denotes words that are not part of a named entity.\n\nUse the specific entity tags: location, person, organization, else and O.\nDataset: WikiNeural.\nSentence: This system was widely copied in various NATO forces .", 
        "prompt_labels": "This(O) system(O) was(O) widely(O) copied(O) in(O) various(O) NATO(B-organization) forces(O) .(O)"
    }, 
    "prediction": "This(O) system(O) was(O) widely(O) copied(O) in(O) various(O) NATO(B-organization) forces(O).(O)"
}
'''
