# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn

from nemo.collections.common.parts import NEG_INF, mask_padded_tokens

__all__ = [
    "GreedySequenceGenerator",
    "TopKSequenceGenerator",
    "BeamSearchSequenceGenerator",
]


class GreedySequenceGenerator(nn.Module):
    """
    Greedy sequence generator based on the decoder followed by log_softmax.

    Args:
        embedding: nn.Module, transforms input_ids into vector embeddings
        decoder: nn.Module, takes embeddings and produces hidden_states
        log_softmax: nn.Module, takes hidden_states and produces log_probs
            which correspond to probability distribution of tokens (ids)
        pad: index of padding token in the vocabulary
        bos: index of beginning of sequence token in the vocabulary
        eos: index of end of sequence token in the vocabulary
        max_sequence_length: maximum allowed length for generated sequences
        max_delta_length: in case of encoder-decoder generation (e.g. NMT),
            forbids generated sequences to be longer than the length of
            source sequences plus max_delta_length
        batch_size: size of the batch of generated sequences if neither
            source nor target starting sequences are provided
    """

    def __init__(
        self,
        embedding,
        decoder,
        log_softmax,
        pad=0,
        bos=1,
        eos=2,
        max_sequence_length=512,
        max_delta_length=20,
        batch_size=1,
    ):
        super().__init__()
        self.embedding = embedding
        self.decoder = decoder
        self.log_softmax = log_softmax
        self.pad, self.bos, self.eos = pad, bos, eos
        self.max_seq_length = max_sequence_length
        self.max_delta_len = max_delta_length
        self.batch_size = batch_size
        # TODO replace self.device attribute with defining device on the fly according to Pytorch Lightning recomendations
        self.device = next(self.decoder.parameters()).device

    @torch.no_grad()
    def _forward(
        self,
        decoder_input_ids=None,
        encoder_hidden_states=None,
        encoder_input_mask=None,
        decoder_mems_list=None,
        pos=0,
    ):
        """
        One step of autoregressive output generation.

        Args:
            decoder_input_ids: starting sequence of tokens to generate from;
                if None, generation will start from a batch of <bos> tokens
            encoder_hidden_states: output of the encoder for conditional
                sequence generation; if None, generator will use unconditional
                mode (e.g., language modeling)
            encoder_input_mask: input mask used in the encoder
            decoder_mems_list: list of size num_layers with cached activations
                of sequence (x[1], ..., x[k-1]) for fast generation of x[k]
            pos: starting position in positional encoding
        """

        decoder_hidden_states = self.embedding.forward(decoder_input_ids, start_pos=pos)
        decoder_input_mask = mask_padded_tokens(decoder_input_ids, self.pad).float()

        if encoder_hidden_states is not None:
            decoder_mems_list = self.decoder.forward(
                decoder_hidden_states,
                decoder_input_mask,
                encoder_hidden_states,
                encoder_input_mask,
                decoder_mems_list,
                return_mems=True,
            )
        else:
            decoder_mems_list = self.decoder.forward(
                decoder_hidden_states, decoder_input_mask, decoder_mems_list, return_mems=True
            )
        log_probs = self.log_softmax.forward(decoder_mems_list[-1])
        return log_probs, decoder_mems_list

    def _prepare_for_search(self, decoder_input_ids=None, encoder_hidden_states=None):
        """
        Helper function which defines starting sequence to begin generating
        with and maximum allowed number of tokens to be generated.
        """

        decoder_parameter = next(self.decoder.parameters())

        batch_size = self.batch_size

        # for encoder-decoder generation, maximum length of generated sequence
        # is min(max_sequence_length, src_len + max_delta_length)
        if encoder_hidden_states is not None:
            batch_size, src_len, _ = encoder_hidden_states.size()
            max_seq_length = min(self.max_seq_length, src_len + self.max_delta_len)
        else:
            max_seq_length = self.max_seq_length

        # if no input is provided, start with the batch of <bos> tokens
        if decoder_input_ids is not None:
            tgt = decoder_input_ids
            batch_size, tgt_len = decoder_input_ids.size()
        else:
            tgt = torch.zeros(batch_size, 1).long().fill_(self.bos).to(decoder_parameter.device)
            tgt_len = 1
        max_generation_length = max_seq_length - tgt_len

        return tgt, batch_size, max_generation_length

    def forward(self, decoder_input_ids=None, encoder_hidden_states=None, encoder_input_mask=None):

        tgt, batch_size, max_generation_length = self._prepare_for_search(decoder_input_ids, encoder_hidden_states)

        # pad profile tracks sequences ending with <eos> token to replace
        # everything after <eos> with <pad> token
        pad_profile = torch.zeros(batch_size, 1).long().to(self.device)

        decoder_mems_list = None
        for i in range(max_generation_length):

            log_probs, decoder_mems_list = self._forward(
                tgt[:, -1:], encoder_hidden_states, encoder_input_mask, decoder_mems_list, i
            )

            next_tokens = torch.argmax(log_probs[:, -1], dim=-1, keepdim=True)
            next_tokens = self.pad * pad_profile + next_tokens * (1 - pad_profile)
            pad_profile = torch.max(pad_profile, (next_tokens == self.eos).long())
            tgt = torch.cat((tgt, next_tokens), dim=-1)

            # abort generation if all sequences end with <eos>
            if pad_profile.sum() == batch_size:
                break

        return tgt


class TopKSequenceGenerator(GreedySequenceGenerator):
    """
    Top-k sequence generator based on the decoder followed by log_softmax.

    Args:
        *all args of GreedySequenceGenerator class
        beam_size: size of the beam (parameter k in top-k)
        temperature: temperature of top-k sampling, all logits are divided
            by temperature before rescaling. High temperature leads to
            uniform distribution, low leads to delta-like distribution.
    Kwargs:
        all remaining parameters of GreedySequenceGenerator class
    """

    def __init__(self, embedding, decoder, log_softmax, beam_size=1, temperature=1.0, **kwargs):
        super().__init__(embedding, decoder, log_softmax, **kwargs)
        self.beam_size = beam_size
        self.temp = temperature

    @torch.no_grad()
    def _forward(
        self,
        decoder_input_ids=None,
        encoder_hidden_states=None,
        encoder_input_mask=None,
        decoder_mems_list=None,
        pos=0,
    ):
        log_probs, decoder_mems_list = super()._forward(
            decoder_input_ids, encoder_hidden_states, encoder_input_mask, decoder_mems_list, pos
        )

        batch_size, seq_len, vocab_size = log_probs.size()
        scores, indices = torch.topk(log_probs, self.beam_size, dim=-1)

        rescaled_logexp = torch.zeros_like(log_probs).scatter(-1, indices, scores.div(self.temp).exp())
        probs = rescaled_logexp / rescaled_logexp.norm(1, -1, keepdim=True)

        # We randomly sample next tokens from rescaled probability distribution
        # over top-k candidates and return a binary tensor which indicates
        # candidates that have been selected. We call this object
        # `pseudo_log_probs` as genuine log_probs should have -infs instead of
        # 0s and 0s instead of 1s.
        ids = torch.multinomial(probs.view(-1, vocab_size), 1).view(-1, seq_len, 1)
        pseudo_log_probs = torch.zeros_like(log_probs).scatter(-1, ids, 1.0)

        return pseudo_log_probs, decoder_mems_list


class BeamSearchSequenceGenerator(GreedySequenceGenerator):
    def __init__(self, embedding, decoder, log_softmax, beam_size=1, len_pen=0, **kwargs):
        """
        Beam Search sequence generator based on the decoder followed by
        log_softmax.

        Args:
            *all args of GreedySequenceGenerator class
            beam_size: size of the beam
            len_pen: length penalty parameter
        Kwargs:
            all remaining parameters of GreedySequenceGenerator class
        """

        super().__init__(embedding, decoder, log_softmax, **kwargs)
        self.beam_size = beam_size
        self.len_pen = len_pen

    @staticmethod
    def compute_len_penalty(lengths, alpha):
        """Returns length penalty according to https://arxiv.org/pdf/1609.08144.pdf"""
        return ((5 + lengths) / 6) ** alpha

    def forward(self, decoder_input_ids=None, encoder_hidden_states=None, encoder_input_mask=None, src_len=None):
        tgt, batch_size, max_generation_length = self._prepare_for_search(decoder_input_ids, encoder_hidden_states)
        if src_len is None:
            src_len = torch.full([batch_size], 512, dtype=torch.int32, device=encoder_hidden_states.device)
        # generate initial buffer of beam_size prefixes-hypotheses
        log_probs_init, decoder_mem_by_layers = self._forward(tgt, encoder_hidden_states, encoder_input_mask, None, 0)
        scores, prefixes = torch.topk(log_probs_init.permute(0, 2, 1), self.beam_size, dim=1)
        scores, prefixes = scores.view(-1, 1), prefixes.view(-1, 1)

        # repeat init target prefixes and cached memory states beam_size times
        prefixes = torch.cat((tgt.repeat(1, self.beam_size).view(-1, 1), prefixes), dim=1)
        for j in range(len(decoder_mem_by_layers)):
            decoder_mem_by_layers[j] = decoder_mem_by_layers[j].repeat(self.beam_size, 1, 1)

        # repeat source sequence beam_size times for beam search
        if encoder_hidden_states is not None:
            _, src_length, hidden_size = encoder_hidden_states.size()
            encoder_input_mask = encoder_input_mask.repeat(1, self.beam_size).view(-1, src_length)
            encoder_hidden_states = encoder_hidden_states.repeat(1, self.beam_size, 1).view(
                -1, src_length, hidden_size
            )
        else:
            hidden_size = decoder_mem_by_layers[0].size(2)

        best_scores = [scores.new_tensor(NEG_INF)] * batch_size
        best_prefixes = [None] * batch_size
        num_remaining_rays = np.full([batch_size], self.beam_size)
        ray_prefixes = list(torch.split(prefixes, tuple(num_remaining_rays)))
        ray_scores = list(torch.split(scores, tuple(num_remaining_rays)))
        # ray_decoder_mem_by_layers = list(
        #     zip(*[torch.split(layer_decoder_mem, num_remaining_rays) for layer_decoder_mem in decoder_mem_by_layers]))
        num_generated_tokens = 1
        while num_remaining_rays.any():
            tgt = torch.cat([p[:, -1:] for p in ray_prefixes])
            print("(BeamSesrchSequenceGenerator.forward)tgt.shape:", tgt.shape)
            # decoder_mem_by_layers_cat = [torch.cat(z) for z in zip(*ray_decoder_mem_by_layers)]
            log_probs_cat, decoder_mem_by_layers= self._forward(
                tgt, encoder_hidden_states, encoder_input_mask, decoder_mem_by_layers, num_generated_tokens)
            print("(BeamSesrchSequenceGenerator.forward)log_probs_cat.shape:", log_probs_cat.shape)
            num_generated_tokens += 1
            ray_log_probs = torch.split(log_probs_cat[:, -1, :], tuple(num_remaining_rays))
            assert decoder_mem_by_layers
            index_dtype = decoder_mem_by_layers[0].topk(0)[1].dtype
            global_living_rays_indices = decoder_mem_by_layers[0].new_tensor(data=[], dtype=index_dtype)
            old_num_rays = 0
            for idx_in_batch in num_remaining_rays.nonzero()[0]:
                token_log_probs, best_tokens_by_ray = torch.topk(
                    ray_log_probs[idx_in_batch], num_remaining_rays[idx_in_batch])
                print("(BeamSesrchSequenceGenerator.forward)ray_scores[idx_in_batch].shape:", ray_scores[idx_in_batch].shape)
                print("(BeamSesrchSequenceGenerator.forward)token_log_probs.shape:", token_log_probs.shape)
                scores_for_batch_elem = (ray_scores[idx_in_batch] + token_log_probs)\
                    .view(num_remaining_rays[idx_in_batch]**2)
                best_scores_for_batch_elem, best_indices = torch.topk(
                    scores_for_batch_elem, num_remaining_rays[idx_in_batch])
                best_ray_indices = best_indices // num_remaining_rays[idx_in_batch]
                best_scores_for_batch_elem = best_scores_for_batch_elem.view(num_remaining_rays[idx_in_batch], 1)
                assert best_ray_indices.dtype == global_living_rays_indices.dtype \
                    and best_ray_indices.device == global_living_rays_indices.device, \
                    f"best_ray_indices.dtype: {best_ray_indices.dtype}\n" \
                    f"global_living_rays_indices.dtype: {global_living_rays_indices.dtype}\n" \
                    f"best_ray_indices.device: {best_ray_indices.device}\n" \
                    f"global_living_rays_indices.device: {global_living_rays_indices.device}\n"
                global_living_rays_indices = torch.cat(
                    [global_living_rays_indices, best_ray_indices + old_num_rays])
                old_num_rays += num_remaining_rays[idx_in_batch]

                best_order_indices = best_indices % num_remaining_rays[idx_in_batch]
                best_tokens_for_batch = best_tokens_by_ray[best_ray_indices, best_order_indices]\
                    .view(num_remaining_rays[idx_in_batch], 1)
                if num_generated_tokens >= max_generation_length \
                        or num_generated_tokens >= src_len[idx_in_batch] + self.max_delta_len:
                    assert num_generated_tokens <= max_generation_length \
                        and num_generated_tokens <= src_len[idx_in_batch] + self.max_delta_len

                    # Add penalty for not finished translation. If translation finished because of length limitation
                    # Add log prob of eos.
                    print("(BeamSesrchSequenceGenerator.forward)best_scores_for_batch_elem.shape:", best_scores_for_batch_elem.shape)
                    print("(BeamSesrchSequenceGenerator.forward)best_tokens_for_batch.shape:", best_tokens_for_batch.shape)
                    best_scores_for_batch_elem += ray_log_probs[idx_in_batch][best_ray_indices, self.eos:self.eos+1] \
                        * (best_tokens_for_batch.ne(self.eos) & best_tokens_for_batch.ne(self.pad))
                    terminated = list(range(num_remaining_rays[idx_in_batch]))
                    living = list()
                else:
                    terminated_mask = best_tokens_for_batch.eq(self.eos) | best_tokens_for_batch.eq(self.pad)
                    print("(BeamSesrchSequenceGenerator.forward)terminated_mask.shape:", terminated_mask.shape)
                    terminated = torch.nonzero(terminated_mask, as_tuple=True)[0]
                    living = torch.nonzero(~terminated_mask, as_tuple=True)[0]
                new_best_ray_index = None
                best_token_i = None
                len_penalty = self.compute_len_penalty(num_generated_tokens, self.len_pen)
                for ti in terminated:
                    candidate_score = best_scores_for_batch_elem[idx_in_batch] / len_penalty
                    if candidate_score > best_scores[idx_in_batch]:
                        best_scores[idx_in_batch] = candidate_score
                        new_best_ray_index = best_ray_indices[ti]
                        best_token_i = ti
                if best_token_i is not None:
                    best_prefixes[idx_in_batch] = torch.cat(
                        [ray_prefixes[idx_in_batch][new_best_ray_index], best_tokens_for_batch[best_token_i]])
                print("(BeamSesrchSequenceGenerator.forward)living:", living)
                print("(BeamSesrchSequenceGenerator.forward)best_scores_for_batch_elem.shape:", best_scores_for_batch_elem.shape)
                ray_scores[idx_in_batch] = best_scores_for_batch_elem[living].view(len(living), 1)
                print("(BeamSesrchSequenceGenerator.forward)best_tokens_for_batch.shape:", best_tokens_for_batch.shape)
                ray_prefixes[idx_in_batch] = torch.cat(
                    [ray_prefixes[idx_in_batch][living], best_tokens_for_batch[living].view(len(living), 1)], dim=1)
                num_remaining_rays[idx_in_batch] = len(living)
            decoder_mem_by_layers = [m[global_living_rays_indices] for m in decoder_mem_by_layers]
            encoder_hidden_states = encoder_hidden_states[global_living_rays_indices]
            encoder_input_mask = encoder_input_mask[global_living_rays_indices]
        return torch.stack(best_prefixes)

    def forward_bak(self, decoder_input_ids=None, encoder_hidden_states=None, encoder_input_mask=None):
        tgt, batch_size, max_generation_length = self._prepare_for_search(decoder_input_ids, encoder_hidden_states)

        # generate initial buffer of beam_size prefixes-hypotheses
        log_probs, decoder_mems_list = self._forward(tgt, encoder_hidden_states, encoder_input_mask, None, 0)
        scores, prefixes = torch.topk(log_probs.permute(0, 2, 1), self.beam_size, dim=1)
        scores, prefixes = scores.view(-1, 1), prefixes.view(-1, 1)

        # repeat init target prefixes and cached memory states beam_size times
        prefixes = torch.cat((tgt.repeat(1, self.beam_size).view(-1, 1), prefixes), dim=1)
        for j in range(len(decoder_mems_list)):
            decoder_mems_list[j] = decoder_mems_list[j].repeat(self.beam_size, 1, 1)
        
        # repeat source sequence beam_size times for beam search
        if encoder_hidden_states is not None:
            _, src_length, hidden_size = encoder_hidden_states.size()
            encoder_input_mask = encoder_input_mask.repeat(1, self.beam_size).view(-1, src_length)
            encoder_hidden_states = encoder_hidden_states.repeat(1, self.beam_size, 1).view(
                -1, src_length, hidden_size
            )
        else:
            hidden_size = decoder_mems_list[0].size(2)

        # pad_profile tracks finished hypotheses to generate only <pad> tokens
        # if <eos> or <pad> has been generated
        pad_profile = torch.zeros_like(scores).long()

        # prefixes_len tracks lengths of generated hypotheses to perform
        # length penalty correction
        prefixes_len = torch.zeros_like(scores).fill_(prefixes.size(1) + 1)

        for i in range(max_generation_length):

            # mask all finished hypotheses to exclude them from beam
            pad_mask = pad_profile.repeat(1, self.beam_size)

            # generate and score candidates for prefixes continuation
            log_probs, decoder_mems_list = self._forward(
                prefixes[:, -1:], encoder_hidden_states, encoder_input_mask, decoder_mems_list, i + 1
            )
            scores_i, prefixes_i = torch.topk(log_probs[:, -1, :], self.beam_size, dim=-1)

            # for all prefixes ending with <eos> or <pad> replace generated
            # continuations with <pad>
            prefixes_i = self.pad * pad_mask + prefixes_i * (1 - pad_mask)

            # force all hypotheses but one generated from already finished
            # hypotheses to have extremely low score, so they will not be
            # considered during beam re-ranking
            pad_mask[:, 1:] = pad_mask[:, 1:] * NEG_INF
            scores = scores + scores_i * (1 - pad_mask).to(scores.dtype)

            # choose top-k hypotheses with length penalty applied
            len_penalties = self.compute_len_penalty(prefixes_len, self.len_pen)
            scores = scores / len_penalties
            scores, indices_i = torch.topk(scores.view(-1, self.beam_size ** 2), self.beam_size, dim=1)
            scores = scores.view(-1, 1) * len_penalties

            # select prefixes which correspond to the chosen hypotheses
            prefixes = prefixes.unsqueeze(1).repeat(1, self.beam_size, 1)
            prefixes = torch.cat((prefixes, prefixes_i.unsqueeze(2)), dim=2)
            prefixes = prefixes.view(batch_size, self.beam_size ** 2, -1)
            p_len = prefixes.size(2)
            prefixes_ids = indices_i.unsqueeze(2).repeat(1, 1, p_len)
            prefixes = prefixes.gather(1, prefixes_ids).view(-1, p_len)

            # reshuffle cached decoder memory states to restore the order
            # of hypotheses broken after top-k selection
            mems_ids = indices_i.unsqueeze(2).unsqueeze(3).repeat(1, 1, p_len - 1, hidden_size) // self.beam_size
            for j in range(len(decoder_mems_list)):
                decoder_mems_list[j] = (
                    decoder_mems_list[j]
                        .view(-1, self.beam_size, p_len - 1, hidden_size)
                        .gather(1, mems_ids)
                        .view(-1, p_len - 1, hidden_size)
                )

            # update prefixes_len and pad_profile
            not_eos_pad = prefixes.ne(self.eos) & prefixes.ne(self.pad)
            prefixes_len = 1 + not_eos_pad.sum(dim=1, keepdim=True).to(scores.dtype)
            pad_profile = (~not_eos_pad[:, -1:]).long()

            # if all hypotheses end with <eos> or <pad>, interrupt search
            if pad_profile.sum() == batch_size * self.beam_size:
                break

        # select best performing hypotheses in each element of the batch
        len_penalties = self.compute_len_penalty(prefixes_len, self.len_pen)
        scores = scores / len_penalties
        best_guesses = (
            torch.argmax(scores.view(-1, self.beam_size), dim=1, keepdim=True).repeat(1, prefixes.size(1)).unsqueeze(1)
        )
        tgt = prefixes.view(batch_size, self.beam_size, -1).gather(1, best_guesses)

        return tgt.squeeze(1)
