from typing import List, NamedTuple

import torch
from pyctcdecode import build_ctcdecoder


from hw_asr.base.base_text_encoder import BaseTextEncoder
from .char_text_encoder import CharTextEncoder
from collections import defaultdict


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, kenlm_model_path: str = None, unigrams_path: str = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        if kenlm_model_path is not None:
            with open(unigrams_path) as f:
                unigrams = [line.strip() for line in f.readlines()]
            self.decoder = build_ctcdecoder(labels=[""] + self.alphabet, kenlm_model_path=kenlm_model_path, unigrams=unigrams)

    def ctc_decode(self, inds: List[int]) -> str:
        # TODO: your code here
        result = []
        last_char = self.EMPTY_TOK
        for ind in inds:
            cur_char = self.ind2char[ind]
            if cur_char != self.EMPTY_TOK and last_char != cur_char:
                result.append(cur_char)
            last_char = cur_char
        return ''.join(result)

    def ctc_beam_search(self, probs: torch.tensor, beam_size: int) -> str:
        """
            Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []
        # TODO: your code here

        def extend_and_merge(frame, state):
            new_state = defaultdict(float)
            for next_char_index, next_char_proba in enumerate(frame):
                for (pref, last_char), pref_proba in state.items():
                    next_char = self.ind2char[next_char_index]
                    if next_char == last_char:
                        new_pref = pref
                    else:
                        if next_char != self.EMPTY_TOK:
                            new_pref = pref + next_char
                        else:
                            new_pref = pref
                        last_char = next_char
                    new_state[(new_pref, last_char)] += pref_proba * next_char_proba
            return new_state

        def truncate(state, beam_size):
            state_list = list(state.items())
            state_list.sort(key=lambda x: -x[1])
            return dict(state_list[:beam_size])

        state = {('', self.EMPTY_TOK): 1.0}
        for frame in probs:
            state = extend_and_merge(frame, state)
            state = truncate(state, beam_size)
        state_list = list(state.items())
        state_list.sort(key=lambda x: -x[1])
        
        # for state in state_list:
        #     hypos.append(Hypothesis(state[0][0], state[1]))
        
        return state_list[0][0][0]
    
    def ctc_lm_beam_search(self, logits: torch.tensor) -> str:
        assert self.decoder is not None
        return self.decoder.decode(logits, beam_width=500).lower()