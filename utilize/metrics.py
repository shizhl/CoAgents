import re
from collections import Counter
import numpy as np
import itertools
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
import spacy
from tqdm import tqdm
from itertools import chain, product
import subprocess

__all__ = ['eval_f1', 'f1_score', 'eval_rouge', 'eval_bleu', 'eval_meteor', 'eval_distinct', 'eval_all',
           'eval_acc', 'eval_em','eval_bert_score']



def eval_all(predicts, answers):
    log_dict = {}
    f1 = eval_f1(predicts, answers)
    log_dict.update(f1)
    rouges = eval_rouge(predicts, answers)
    log_dict.update(rouges)
    bleu = eval_bleu(predicts, answers)
    log_dict.update(bleu)
    meteors = eval_meteor(predicts, answers)
    log_dict.update(meteors)
    distinct = eval_redial_dist(predicts, answers)
    log_dict.update(distinct)
    # embedding = eval_embedding(predicts, answers)
    # log_dict.update(embedding)
    return log_dict

def eval_bert_score(predictions,references):
    from evaluate import load
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    results={
        "bs-p":sum(results['precision'])/len(results['precision']),
        "bs-r":sum(results['recall'])/len(results['recall']),
    }
    results['bs-f1']=2*results['bs-p']*results['bs-r']/(results['bs-p']+results['bs-r'])
    return results

def eval_acc(predictions, raw, cache=None, compute_cache=False):
    def lower(text):
        if isinstance(text, str):
            text = text.strip().lower()
            text = ' '.join(nltk.word_tokenize(text))
            return text.strip()
        return [lower(item) for item in text]
    nlp = spacy.load("en_core_web_sm")
    ent_f1 = []
    k_f1 = []
    sent_acc = []
    build_cache = []
    if cache is not None:
        raw = cache
    if compute_cache:
        predictions = tqdm(raw)

    for pred, example in zip(predictions, raw):
        if cache is not None:
            label_knowledge, label_response, label_ents, all_candidates = example
        else:
            if isinstance(example['title'], list):
                label_knowledge = [lower(f'{t} {s}') for t, s in zip(example['title'], example['checked_sentence'])]
            else:
                label_knowledge = [lower(example['title'] + ' ' + example['checked_sentence'])]
            label_response = lower(example['labels'][0])
            label_ents = [ent.text for ent in nlp(label_response).ents]
            all_candidates = [lower(f'{title} {sentence}') for title in example['knowledge'] for sentence in
                              example['knowledge'][title]]
        if compute_cache:
            build_cache.append([label_knowledge, label_response, label_ents, all_candidates])
        else:
            pred_response = lower(pred)
            pred_ents = [ent.text for ent in nlp(pred_response).ents]
            if len(label_ents) > 0:
                ent_f1.append(f1_score(' '.join(pred_ents), [' '.join(label_ents)]))
            if len(label_knowledge) == 0:
                k_f1.append(0)
            else:
                k_f1.append(f1_score(pred_response, label_knowledge))
            max_candidates_f1 = max([f1_score(sent, [pred_response]) for sent in all_candidates])
            sent_acc.append(int(max_candidates_f1 == k_f1[-1]))
    if compute_cache:
        return build_cache
    return {'KF1': sum(k_f1) / len(k_f1) * 100,
            'EntF1': sum(ent_f1) / len(ent_f1) * 100,
            'ACC': sum(sent_acc) / len(sent_acc) * 100}


def eval_em(predictions, raw):
    import re
    import string
    def norm_text(s):
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

    scores = []
    for pred, example in zip(predictions, raw):
        pred = norm_text(pred)
        if any([norm_text(ans) in pred for ans in example]):
            scores.append(1)
        else:
            scores.append(0)
    return {'NQ-EM': sum(scores) / len(scores)}



re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def rounder(num):
    return round(num, 2)


# ==================================================================================================
# F1 Score
# ==================================================================================================

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _prec_recall_f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.

    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values

    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def f1_score(guess, answers):
    if guess is None or answers is None:
        return 0
    g_tokens = normalize_answer(guess).split()
    scores = [
        _prec_recall_f1_score(g_tokens, normalize_answer(a).split()) for a in answers
    ]
    return max(f1 for p, r, f1 in scores)


def eval_f1(predicts, answers):
    f1 = 0.
    for predict, answer in zip(predicts, answers):
        answer = answer.split('\t')
        f1 += f1_score(predict, answer)
    return {'F1': rounder(f1 * 100 / len(answers))}


# ==================================================================================================
# ROUGE Score
# ==================================================================================================

def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _split_into_words(sentences):
    """Splits multiple sentences into words and flattens the result"""
    return list(itertools.chain(*[_.split(" ") for _ in sentences]))


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    words = _split_into_words(sentences)
    return _get_ngrams(n, words)


def _len_lcs(x, y):
    """
    Returns the length of the Longest Common Subsequence between sequences x
    and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: sequence of words
      y: sequence of words

    Returns
      integer: Length of LCS between x and y
    """
    table = _lcs(x, y)
    n, m = len(x), len(y)
    return table[n, m]


def _lcs(x, y):
    """
    Computes the length of the longest common subsequence (lcs) between two
    strings. The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: collection of words
      y: collection of words

    Returns:
      Table of dictionary of coord and len lcs
    """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def _recon_lcs(x, y):
    """
    Returns the Longest Subsequence between x and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: sequence of words
      y: sequence of words

    Returns:
      sequence: LCS of x and y
    """
    i, j = len(x), len(y)
    table = _lcs(x, y)

    def _recon(i, j):
        """private recon calculation"""
        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [(x[i - 1], i)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)

    recon_tuple = tuple(map(lambda x: x[0], _recon(i, j)))
    return recon_tuple


def rouge_n(evaluated_sentences, reference_sentences, n=2):
    """
    Computes ROUGE-N of two text collections of sentences.
    Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
    papers/rouge-working-note-v1.3.1.pdf

    Args:
      evaluated_sentences: The sentences that have been picked by the summarizer
      reference_sentences: The sentences from the referene set
      n: Size of ngram.  Defaults to 2.

    Returns:
      A tuple (f1, precision, recall) for ROUGE-N

    Raises:
      ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = _get_word_ngrams(n, reference_sentences)
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    # Gets the overlapping ngrams between evaluated and reference
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    # Handle edge case. This isn't mathematically correct, but it's good enough
    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

    # return overlapping_count / reference_count
    return f1_score, precision, recall


def _f_p_r_lcs(llcs, m, n):
    """
    Computes the LCS-based F-measure score
    Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf

    Args:
      llcs: Length of LCS
      m: number of words in reference summary
      n: number of words in candidate summary

    Returns:
      Float. LCS-based F-measure score
    """
    r_lcs = llcs / m
    p_lcs = llcs / n
    beta = p_lcs / (r_lcs + 1e-12)
    num = (1 + (beta ** 2)) * r_lcs * p_lcs
    denom = r_lcs + ((beta ** 2) * p_lcs)
    f_lcs = num / (denom + 1e-12)
    return f_lcs, p_lcs, r_lcs


def rouge_l_sentence_level(evaluated_sentences, reference_sentences):
    """
    Computes ROUGE-L (sentence level) of two text collections of sentences.
    http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf

    Calculated according to:
    R_lcs = LCS(X,Y)/m
    P_lcs = LCS(X,Y)/n
    F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

    where:
    X = reference summary
    Y = Candidate summary
    m = length of reference summary
    n = length of candidate summary

    Args:
      evaluated_sentences: The sentences that have been picked by the summarizer
      reference_sentences: The sentences from the referene set

    Returns:
      A float: F_lcs

    Raises:
      ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")
    reference_words = _split_into_words(reference_sentences)
    evaluated_words = _split_into_words(evaluated_sentences)
    m = len(reference_words)
    n = len(evaluated_words)
    lcs = _len_lcs(evaluated_words, reference_words)
    return _f_p_r_lcs(lcs, m, n)


def _union_lcs(evaluated_sentences, reference_sentence):
    """
    Returns LCS_u(r_i, C) which is the LCS score of the union longest common
    subsequence between reference sentence ri and candidate summary C. For example
    if r_i= w1 w2 w3 w4 w5, and C contains two sentences: c1 = w1 w2 w6 w7 w8 and
    c2 = w1 w3 w8 w9 w5, then the longest common subsequence of r_i and c1 is
    “w1 w2” and the longest common subsequence of r_i and c2 is “w1 w3 w5”. The
    union longest common subsequence of r_i, c1, and c2 is “w1 w2 w3 w5” and
    LCS_u(r_i, C) = 4/5.

    Args:
      evaluated_sentences: The sentences that have been picked by the summarizer
      reference_sentence: One of the sentences in the reference summaries

    Returns:
      float: LCS_u(r_i, C)

    ValueError:
      Raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    lcs_union = set()
    reference_words = _split_into_words([reference_sentence])
    combined_lcs_length = 0
    for eval_s in evaluated_sentences:
        evaluated_words = _split_into_words([eval_s])
        lcs = set(_recon_lcs(reference_words, evaluated_words))
        combined_lcs_length += len(lcs)
        lcs_union = lcs_union.union(lcs)

    union_lcs_count = len(lcs_union)
    union_lcs_value = union_lcs_count / combined_lcs_length
    return union_lcs_value


def rouge_l_summary_level(evaluated_sentences, reference_sentences):
    """
    Computes ROUGE-L (summary level) of two text collections of sentences.
    http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf

    Calculated according to:
    R_lcs = SUM(1, u)[LCS<union>(r_i,C)]/m
    P_lcs = SUM(1, u)[LCS<union>(r_i,C)]/n
    F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

    where:
    SUM(i,u) = SUM from i through u
    u = number of sentences in reference summary
    C = Candidate summary made up of v sentences
    m = number of words in reference summary
    n = number of words in candidate summary

    Args:
      evaluated_sentences: The sentences that have been picked by the summarizer
      reference_sentence: One of the sentences in the reference summaries

    Returns:
      A float: F_lcs

    Raises:
      ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    # total number of words in reference sentences
    m = len(_split_into_words(reference_sentences))

    # total number of words in evaluated sentences
    n = len(_split_into_words(evaluated_sentences))

    union_lcs_sum_across_all_references = 0
    for ref_s in reference_sentences:
        union_lcs_sum_across_all_references += _union_lcs(evaluated_sentences,
                                                          ref_s)
    return _f_p_r_lcs(union_lcs_sum_across_all_references, m, n)


def rouge(hypotheses, references):
    """Calculates average rouge scores for a list of hypotheses and
    references"""

    # Filter out hyps that are of 0 length
    # hyps_and_refs = zip(hypotheses, references)
    # hyps_and_refs = [_ for _ in hyps_and_refs if len(_[0]) > 0]
    # hypotheses, references = zip(*hyps_and_refs)

    # Calculate ROUGE-1 F1, precision, recall scores
    rouge_1 = [
        rouge_n([hyp], [ref], 1) for hyp, ref in zip(hypotheses, references)
    ]
    rouge_1_f, rouge_1_p, rouge_1_r = map(np.mean, zip(*rouge_1))

    # Calculate ROUGE-2 F1, precision, recall scores
    rouge_2 = [
        rouge_n([hyp], [ref], 2) for hyp, ref in zip(hypotheses, references)
    ]
    rouge_2_f, rouge_2_p, rouge_2_r = map(np.mean, zip(*rouge_2))

    # Calculate ROUGE-L F1, precision, recall scores
    rouge_l = [
        rouge_l_sentence_level([hyp], [ref])
        for hyp, ref in zip(hypotheses, references)
    ]
    rouge_l_f, rouge_l_p, rouge_l_r = map(np.mean, zip(*rouge_l))

    return {
        "rouge_1/f_score": rouge_1_f,
        "rouge_1/r_score": rouge_1_r,
        "rouge_1/p_score": rouge_1_p,
        "rouge_2/f_score": rouge_2_f,
        "rouge_2/r_score": rouge_2_r,
        "rouge_2/p_score": rouge_2_p,
        "rouge_l/f_score": rouge_l_f,
        "rouge_l/r_score": rouge_l_r,
        "rouge_l/p_score": rouge_l_p,
    }


def cal_rouge(run, ref):
    x = rouge(run, ref)
    return x['rouge_1/f_score'] * 100, x['rouge_2/f_score'] * 100, x['rouge_l/f_score'] * 100


def rouge_max_over_ground_truths(prediction, ground_truths):
    scores_for_rouge1 = []
    scores_for_rouge2 = []
    scores_for_rougel = []
    for ground_truth in ground_truths:
        score = cal_rouge([prediction], [ground_truth])
        scores_for_rouge1.append(score[0])
        scores_for_rouge2.append(score[1])
        scores_for_rougel.append(score[2])
    return max(scores_for_rouge1), max(scores_for_rouge2), max(scores_for_rougel)


def _eval_rouge(run, ref):
    rouge_1 = rouge_2 = rouge_l = total = 0
    assert len(run) == len(ref), "the length of predicted span and ground_truths span should be same"

    for i, pre in enumerate(run):
        rouge_result = rouge_max_over_ground_truths(pre, ref[i])
        rouge_1 += rouge_result[0]
        rouge_2 += rouge_result[1]
        rouge_l += rouge_result[2]
        total += 1

    rouge_1 = rouge_1 / total
    rouge_2 = rouge_2 / total
    rouge_l = rouge_l / total

    return {'ROUGE_1_F1': rounder(rouge_1), 'ROUGE_2_F1': rounder(rouge_2), 'ROUGE_L_F1': rounder(rouge_l)}


def eval_rouge(predicts, answers):
    _answers = []
    for answer in answers:
        answer = answer.split('\t')
        _answers.append(answer)
    return _eval_rouge(predicts, _answers)


# ==================================================================================================
# BLEU Score
# ==================================================================================================

def eval_bleu(predicts, answers):
    _answers = []
    for answer in answers:
        answer = answer.split('\t')
        _answers.append([item.split(' ') for item in answer])
    bleu1 = 0.
    bleu2 = 0.
    bleu3 = 0.
    bleu4 = 0.

    for predict, answer in zip(predicts, _answers):  # run_dict是example的数量
        predict = predict.split(' ')
        bleu1 += sentence_bleu(answer, predict, weights=[1, 0, 0, 0])
        bleu2 += sentence_bleu(answer, predict, weights=[0.5, 0.5, 0, 0])
        bleu3 += sentence_bleu(answer, predict, weights=[1 / 3, 1 / 3, 1 / 3, 0])
        bleu4 += sentence_bleu(answer, predict, weights=[0.25, 0.25, 0.25, 0.25])
    return {'BLEU1': rounder(bleu1 * 100 / len(_answers)), 'BLEU2': rounder(bleu2 * 100 / len(_answers)),
            'BLEU3': rounder(bleu3 * 100 / len(_answers)), 'BLEU4': rounder(bleu4 * 100 / len(_answers))}


# ==================================================================================================
# METEOR Score
# ==================================================================================================

def _generate_enums(hypothesis, reference, preprocess=str.lower):
    """
    Takes in string inputs for hypothesis and reference and returns
    enumerated word lists for each of them

    :param hypothesis: hypothesis string
    :type hypothesis: str
    :param reference: reference string
    :type reference: str
    :preprocess: preprocessing method (default str.lower)
    :type preprocess: method
    :return: enumerated words list
    :rtype: list of 2D tuples, list of 2D tuples
    """
    hypothesis_list = list(enumerate(preprocess(hypothesis).split()))
    reference_list = list(enumerate(preprocess(reference).split()))
    return hypothesis_list, reference_list


def exact_match(hypothesis, reference):
    """
    matches exact words in hypothesis and reference
    and returns a word mapping based on the enumerated
    word id between hypothesis and reference

    :param hypothesis: hypothesis string
    :type hypothesis: str
    :param reference: reference string
    :type reference: str
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    hypothesis_list, reference_list = _generate_enums(hypothesis, reference)
    return _match_enums(hypothesis_list, reference_list)


def _match_enums(enum_hypothesis_list, enum_reference_list):
    """
    matches exact words in hypothesis and reference and returns
    a word mapping between enum_hypothesis_list and enum_reference_list
    based on the enumerated word id.

    :param enum_hypothesis_list: enumerated hypothesis list
    :type enum_hypothesis_list: list of tuples
    :param enum_reference_list: enumerated reference list
    :type enum_reference_list: list of 2D tuples
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    word_match = []
    for i in range(len(enum_hypothesis_list))[::-1]:
        for j in range(len(enum_reference_list))[::-1]:
            if enum_hypothesis_list[i][1] == enum_reference_list[j][1]:
                word_match.append((enum_hypothesis_list[i][0], enum_reference_list[j][0]))
                (enum_hypothesis_list.pop(i)[1], enum_reference_list.pop(j)[1])
                break
    return word_match, enum_hypothesis_list, enum_reference_list


def _enum_stem_match(enum_hypothesis_list, enum_reference_list, stemmer=PorterStemmer()):
    """
    Stems each word and matches them in hypothesis and reference
    and returns a word mapping between enum_hypothesis_list and
    enum_reference_list based on the enumerated word id. The function also
    returns a enumerated list of unmatched words for hypothesis and reference.

    :param enum_hypothesis_list:
    :type enum_hypothesis_list:
    :param enum_reference_list:
    :type enum_reference_list:
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    stemmed_enum_list1 = [(word_pair[0], stemmer.stem(word_pair[1])) \
                          for word_pair in enum_hypothesis_list]

    stemmed_enum_list2 = [(word_pair[0], stemmer.stem(word_pair[1])) \
                          for word_pair in enum_reference_list]

    word_match, enum_unmat_hypo_list, enum_unmat_ref_list = \
        _match_enums(stemmed_enum_list1, stemmed_enum_list2)

    enum_unmat_hypo_list = list(zip(*enum_unmat_hypo_list)) if len(enum_unmat_hypo_list) > 0 else []

    enum_unmat_ref_list = list(zip(*enum_unmat_ref_list)) if len(enum_unmat_ref_list) > 0 else []

    enum_hypothesis_list = list(filter(lambda x: x[0] not in enum_unmat_hypo_list,
                                       enum_hypothesis_list))

    enum_reference_list = list(filter(lambda x: x[0] not in enum_unmat_ref_list,
                                      enum_reference_list))

    return word_match, enum_hypothesis_list, enum_reference_list


def stem_match(hypothesis, reference, stemmer=PorterStemmer()):
    """
    Stems each word and matches them in hypothesis and reference
    and returns a word mapping between hypothesis and reference

    :param hypothesis:
    :type hypothesis:
    :param reference:
    :type reference:
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that
                   implements a stem method
    :return: enumerated matched tuples, enumerated unmatched hypothesis tuples,
             enumerated unmatched reference tuples
    :rtype: list of 2D tuples, list of 2D tuples,  list of 2D tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_stem_match(enum_hypothesis_list, enum_reference_list, stemmer=stemmer)


def _enum_wordnetsyn_match(enum_hypothesis_list, enum_reference_list, wordnet=wordnet):
    """
    Matches each word in reference to a word in hypothesis
    if any synonym of a hypothesis word is the exact match
    to the reference word.

    :param enum_hypothesis_list: enumerated hypothesis list
    :param enum_reference_list: enumerated reference list
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: list of matched tuples, unmatched hypothesis list, unmatched reference list
    :rtype:  list of tuples, list of tuples, list of tuples

    """
    word_match = []
    for i in range(len(enum_hypothesis_list))[::-1]:
        hypothesis_syns = set(chain(*[[lemma.name() for lemma in synset.lemmas()
                                       if lemma.name().find('_') < 0]
                                      for synset in \
                                      wordnet.synsets(
                                          enum_hypothesis_list[i][1])]
                                    )).union({enum_hypothesis_list[i][1]})
        for j in range(len(enum_reference_list))[::-1]:
            if enum_reference_list[j][1] in hypothesis_syns:
                word_match.append((enum_hypothesis_list[i][0], enum_reference_list[j][0]))
                enum_hypothesis_list.pop(i), enum_reference_list.pop(j)
                break
    return word_match, enum_hypothesis_list, enum_reference_list


def wordnetsyn_match(hypothesis, reference, wordnet=wordnet):
    """
    Matches each word in reference to a word in hypothesis if any synonym
    of a hypothesis word is the exact match to the reference word.

    :param hypothesis: hypothesis string
    :param reference: reference string
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: list of mapped tuples
    :rtype: list of tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_wordnetsyn_match(enum_hypothesis_list, enum_reference_list, wordnet=wordnet)


def _enum_allign_words(enum_hypothesis_list, enum_reference_list,
                       stemmer=PorterStemmer(), wordnet=wordnet):
    """
    Aligns/matches words in the hypothesis to reference by sequentially
    applying exact match, stemmed match and wordnet based synonym match.
    in case there are multiple matches the match which has the least number
    of crossing is chosen. Takes enumerated list as input instead of
    string input

    :param enum_hypothesis_list: enumerated hypothesis list
    :param enum_reference_list: enumerated reference list
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: sorted list of matched tuples, unmatched hypothesis list,
             unmatched reference list
    :rtype: list of tuples, list of tuples, list of tuples
    """
    exact_matches, enum_hypothesis_list, enum_reference_list = \
        _match_enums(enum_hypothesis_list, enum_reference_list)

    stem_matches, enum_hypothesis_list, enum_reference_list = \
        _enum_stem_match(enum_hypothesis_list, enum_reference_list,
                         stemmer=stemmer)

    wns_matches, enum_hypothesis_list, enum_reference_list = \
        _enum_wordnetsyn_match(enum_hypothesis_list, enum_reference_list,
                               wordnet=wordnet)

    return (sorted(exact_matches + stem_matches + wns_matches, key=lambda wordpair: wordpair[0]),
            enum_hypothesis_list, enum_reference_list)


def allign_words(hypothesis, reference, stemmer=PorterStemmer(), wordnet=wordnet):
    """
    Aligns/matches words in the hypothesis to reference by sequentially
    applying exact match, stemmed match and wordnet based synonym match.
    In case there are multiple matches the match which has the least number
    of crossing is chosen.

    :param hypothesis: hypothesis string
    :param reference: reference string
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :return: sorted list of matched tuples, unmatched hypothesis list, unmatched reference list
    :rtype: list of tuples, list of tuples, list of tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_allign_words(enum_hypothesis_list, enum_reference_list, stemmer=stemmer,
                              wordnet=wordnet)


def _count_chunks(matches):
    """
    Counts the fewest possible number of chunks such that matched unigrams
    of each chunk are adjacent to each other. This is used to caluclate the
    fragmentation part of the metric.

    :param matches: list containing a mapping of matched words (output of allign_words)
    :return: Number of chunks a sentence is divided into post allignment
    :rtype: int
    """
    i = 0
    chunks = 1
    while (i < len(matches) - 1):
        if (matches[i + 1][0] == matches[i][0] + 1) and (matches[i + 1][1] == matches[i][1] + 1):
            i += 1
            continue
        i += 1
        chunks += 1
    return chunks


def single_meteor_score(reference,
                        hypothesis,
                        preprocess=str.lower,
                        stemmer=PorterStemmer(),
                        wordnet=wordnet,
                        alpha=0.9,
                        beta=3,
                        gamma=0.5):
    """
    Calculates METEOR score for single hypothesis and reference as per
    "Meteor: An Automatic Metric for MT Evaluation with HighLevels of
    Correlation with Human Judgments" by Alon Lavie and Abhaya Agarwal,
    in Proceedings of ACL.
    http://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf


    >>> hypothesis1 = 'It is a guide to action which ensures that the military always obeys the commands of the party'

    >>> reference1 = 'It is a guide to action that ensures that the military will forever heed Party commands'


    >>> round(single_meteor_score(reference1, hypothesis1),4)
    0.7398

        If there is no words match during the alignment the method returns the
        score as 0. We can safely  return a zero instead of raising a
        division by zero error as no match usually implies a bad translation.

    >>> round(meteor_score('this is a cat', 'non matching hypothesis'),4)
    0.0

    :param references: reference sentences
    :type references: list(str)
    :param hypothesis: a hypothesis sentence
    :type hypothesis: str
    :param preprocess: preprocessing function (default str.lower)
    :type preprocess: method
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :param alpha: parameter for controlling relative weights of precision and recall.
    :type alpha: float
    :param beta: parameter for controlling shape of penalty as a
                 function of as a function of fragmentation.
    :type beta: float
    :param gamma: relative weight assigned to fragmentation penality.
    :type gamma: float
    :return: The sentence-level METEOR score.
    :rtype: float
    """
    enum_hypothesis, enum_reference = _generate_enums(hypothesis,
                                                      reference,
                                                      preprocess=preprocess)
    translation_length = len(enum_hypothesis)
    reference_length = len(enum_reference)
    matches, _, _ = _enum_allign_words(enum_hypothesis, enum_reference)
    matches_count = len(matches)
    try:
        precision = float(matches_count) / translation_length
        recall = float(matches_count) / reference_length
        fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        chunk_count = float(_count_chunks(matches))
        frag_frac = chunk_count / matches_count
    except ZeroDivisionError:
        return 0.0
    penalty = gamma * frag_frac ** beta
    return (1 - penalty) * fmean


def meteor_score(references,
                 hypothesis,
                 preprocess=str.lower,
                 stemmer=PorterStemmer(),
                 wordnet=wordnet,
                 alpha=0.9,
                 beta=3,
                 gamma=0.5):
    """
    Calculates METEOR score for hypothesis with multiple references as
    described in "Meteor: An Automatic Metric for MT Evaluation with
    HighLevels of Correlation with Human Judgments" by Alon Lavie and
    Abhaya Agarwal, in Proceedings of ACL.
    http://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf


    In case of multiple references the best score is chosen. This method
    iterates over single_meteor_score and picks the best pair among all
    the references for a given hypothesis

    >>> hypothesis1 = 'It is a guide to action which ensures that the military always obeys the commands of the party'
    >>> hypothesis2 = 'It is to insure the troops forever hearing the activity guidebook that party direct'

    >>> reference1 = 'It is a guide to action that ensures that the military will forever heed Party commands'
    >>> reference2 = 'It is the guiding principle which guarantees the military forces always being under the command of the Party'
    >>> reference3 = 'It is the practical guide for the army always to heed the directions of the party'

    >>> round(meteor_score([reference1, reference2, reference3], hypothesis1),4)
    0.7398

        If there is no words match during the alignment the method returns the
        score as 0. We can safely  return a zero instead of raising a
        division by zero error as no match usually implies a bad translation.

    >>> round(meteor_score(['this is a cat'], 'non matching hypothesis'),4)
    0.0

    :param references: reference sentences
    :type references: list(str)
    :param hypothesis: a hypothesis sentence
    :type hypothesis: str
    :param preprocess: preprocessing function (default str.lower)
    :type preprocess: method
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :type stemmer: nltk.stem.api.StemmerI or any class that implements a stem method
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :type wordnet: WordNetCorpusReader
    :param alpha: parameter for controlling relative weights of precision and recall.
    :type alpha: float
    :param beta: parameter for controlling shape of penalty as a function
                 of as a function of fragmentation.
    :type beta: float
    :param gamma: relative weight assigned to fragmentation penality.
    :type gamma: float
    :return: The sentence-level METEOR score.
    :rtype: float
    """
    return max([single_meteor_score(reference,
                                    hypothesis,
                                    stemmer=stemmer,
                                    wordnet=wordnet,
                                    alpha=alpha,
                                    beta=beta,
                                    gamma=gamma) for reference in references])


def eval_meteor(predicts, answers):
    meteor = 0.
    for predict, answer in zip(predicts, answers):
        answer = answer.split('\t')
        meteor += meteor_score(answer, predict)
    return {'METEOR': rounder(meteor * 100 / len(answers))}


# ==================================================================================================
# DISTINCT Score
# ==================================================================================================


def eval_distinct(predicts, answers=None):
    hypotheses = []
    for predict in predicts:
        hypotheses.append(predict.split(' '))
    hypothesis_pipe = '\n'.join([' '.join(hyp) for hyp in hypotheses])
    pipe = subprocess.Popen(["perl", 'diversity.pl.remove_extension'], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    pipe.stdin.write(hypothesis_pipe.encode())
    pipe.stdin.close()
    diversity = pipe.stdout.read()
    diversity = str(diversity).strip().split()
    diver_uni = float(diversity[0][2:])
    diver_bi = float(diversity[1][:-3])
    return {'Distinct-1': rounder(diver_uni * 100), 'Distinct-2': rounder(diver_bi * 100)}


def generate_n_grams(x, n):
    n_grams = set(zip(*[x[i:] for i in range(n)]))
    # print(x, n_grams)
    # for n_gram in n_grams:
    #     x.append(' '.join(n_gram))
    return n_grams


def distinct_n_grams(tokenized_lines, n):
    n_grams_all = set()
    for line in tokenized_lines:
        n_grams = generate_n_grams(line, n)
        # print(line, n_grams)
        n_grams_all |= n_grams
    total_len = 0
    for item in tokenized_lines:
        total_len += len(item)
    return len(set(n_grams_all)), len(set(n_grams_all)) / max(total_len, 1)  # len(tokenized_lines)


def eval_redial_dist(predicts, answers):
    predicts = [item.split() for item in predicts]
    _, dist2 = distinct_n_grams(predicts, 2)
    _, dist3 = distinct_n_grams(predicts, 3)
    return {'ReDist2': rounder(dist2 * 100), 'ReDist3': rounder(dist3 * 100)}

# ==================================================================================================
# EMBEDDING Score
# ==================================================================================================


# def eval_embedding(predicts, answers):
#     from nlgeval import NLGEval
#     nlgeval = NLGEval(no_overlap=True, no_skipthoughts=True)  # loads the models
#     metrics_dict = nlgeval.compute_metrics([answers], predicts)
#     metrics = {'Average': rounder(metrics_dict['EmbeddingAverageCosineSimilarity'] * 100),
#                'Extrema': rounder(metrics_dict['VectorExtremaCosineSimilarity'] * 100),
#                'Greedy': rounder(metrics_dict['GreedyMatchingScore'] * 100)}
#     return metrics

