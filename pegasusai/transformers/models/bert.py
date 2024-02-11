from typing import Dict, List, Optional, Union

import transformers
from datasets import Dataset, Features
from transformers.models.bert import BertTokenizerFast

from pegasusai.datasets import dataset_dict_map


def bert_tokenizer_fast___call__(
    input_tokenizer: BertTokenizerFast,
    text: Union[str, List[str], List[List[str]]] = None,
    text_pair: Union[str, List[str], List[List[str]], None] = None,
    text_target: Union[str, List[str], List[List[str]]] = None,
    text_pair_target: Union[str, List[str], List[List[str]], None] = None,
    add_special_tokens: bool = True,
    padding: Union[bool, str, transformers.utils.generic.PaddingStrategy] = False,
    truncation: Union[
        bool, str, transformers.tokenization_utils_base.TruncationStrategy
    ] = None,
    max_length: Optional[int] = None,
    stride: int = 0,
    is_split_into_words: bool = False,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Union[str, transformers.utils.generic.TensorType, None] = None,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
) -> transformers.tokenization_utils_base.BatchEncoding:
    return input_tokenizer.__call__(
        text=text,
        text_pair=text_pair,
        text_target=text_target,
        text_pair_target=text_pair_target,
        add_special_tokens=add_special_tokens,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        stride=stride,
        is_split_into_words=is_split_into_words,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors=return_tensors,
        return_token_type_ids=return_token_type_ids,
        return_attention_mask=return_attention_mask,
        return_overflowing_tokens=return_overflowing_tokens,
        return_special_tokens_mask=return_special_tokens_mask,
        return_offsets_mapping=return_offsets_mapping,
        return_length=return_length,
        verbose=verbose,
    )


def bert_tokenizer_fast___init__(
    input_tokenizer: BertTokenizerFast,
    vocab_file=None,
    tokenizer_file=None,
    do_lower_case=True,
    unk_token="[UNK]",
    sep_token="[SEP]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    mask_token="[MASK]",
    tokenize_chinese_chars=True,
    strip_accents=None,
    **kwargs
) -> None:
    return input_tokenizer.__init__(
        vocab_file=vocab_file,
        tokenizer_file=tokenizer_file,
        do_lower_case=do_lower_case,
        unk_token=unk_token,
        sep_token=sep_token,
        pad_token=pad_token,
        cls_token=cls_token,
        mask_token=mask_token,
        tokenize_chinese_chars=tokenize_chinese_chars,
        strip_accents=strip_accents,
        kwargs=kwargs,
    )


def bert_tokenizer_fast_tokenize_dataset_dict(
    input_tokenizer: BertTokenizerFast,
    input_dataset_dict: Dataset,
    tokenizer_target_column: str,
    text_pair: Union[str, List[str], List[List[str]], None] = None,
    text_target: Union[str, List[str], List[List[str]]] = None,
    text_pair_target: Union[str, List[str], List[List[str]], None] = None,
    add_special_tokens: bool = True,
    padding: Union[bool, str, transformers.utils.generic.PaddingStrategy] = False,
    truncation: Union[
        bool, str, transformers.tokenization_utils_base.TruncationStrategy
    ] = None,
    max_length: Optional[int] = None,
    stride: int = 0,
    is_split_into_words: bool = False,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Union[str, transformers.utils.generic.TensorType, None] = None,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
    with_indices: bool = False,
    with_rank: bool = False,
    input_columns: Union[str, List[str], None] = None,
    batched: bool = False,
    batch_size: Optional[int] = 1000,
    drop_last_batch: bool = False,
    remove_columns: Union[str, List[str], None] = None,
    keep_in_memory: bool = False,
    load_from_cache_file: Optional[bool] = None,
    cache_file_names: Optional[Dict[str, Optional[str]]] = None,
    writer_batch_size: Optional[int] = 1000,
    features: Optional[Features] = None,
    disable_nullable: bool = False,
    fn_kwargs: Optional[dict] = None,
    num_proc: Optional[int] = None,
    desc: Optional[str] = None,
):
    def tokenize_function(examples):
        return bert_tokenizer_fast___call__(
            input_tokenizer=input_tokenizer,
            text=examples[tokenizer_target_column],
            text_pair=text_pair,
            text_target=text_target,
            text_pair_target=text_pair_target,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
        )

    return dataset_dict_map(
        input_dataset_dict=input_dataset_dict,
        function=tokenize_function,
        with_indices=with_indices,
        with_rank=with_rank,
        input_columns=input_columns,
        batched=batched,
        batch_size=batch_size,
        drop_last_batch=drop_last_batch,
        remove_columns=remove_columns,
        keep_in_memory=keep_in_memory,
        load_from_cache_file=load_from_cache_file,
        cache_file_names=cache_file_names,
        writer_batch_size=writer_batch_size,
        features=features,
        disable_nullable=disable_nullable,
        fn_kwargs=fn_kwargs,
        num_proc=num_proc,
        desc=desc,
    )
