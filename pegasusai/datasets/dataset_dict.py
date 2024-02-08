import os
from typing import Callable, Dict, List, Optional, Sequence, Union

import datasets
import huggingface_hub
import numpy as np
from datasets import DatasetDict


def dataset_dict___enter__(input_dataset_dict: DatasetDict) -> None:
    return input_dataset_dict.__enter__()


def dataset_dict___exit__(
    input_dataset_dict: DatasetDict, exc_type, exc_val, exc_tb
) -> None:
    return input_dataset_dict.__exit__(
        exc_type=exc_type,
        exc_val=exc_val,
        exc_tb=exc_tb,
    )


def dataset_dict___getitem__(
    input_dataset_dict: DatasetDict, k
) -> datasets.arrow_dataset.Dataset:
    return input_dataset_dict.__getitem__(
        k=k,
    )


def dataset_dict___repr__(input_dataset_dict: DatasetDict) -> None:
    return input_dataset_dict.__repr__()


def dataset_dict__check_values_features(input_dataset_dict: DatasetDict) -> None:
    return input_dataset_dict._check_values_features()


def dataset_dict__check_values_type(input_dataset_dict: DatasetDict) -> None:
    return input_dataset_dict._check_values_type()


def dataset_dict_align_labels_with_mapping(
    input_dataset_dict: DatasetDict, label2id: Dict, label_column: str
) -> "DatasetDict":
    return input_dataset_dict.align_labels_with_mapping(
        label2id=label2id,
        label_column=label_column,
    )


def dataset_dict_cast(
    input_dataset_dict: DatasetDict, features: datasets.features.features.Features
) -> "DatasetDict":
    return input_dataset_dict.cast(
        features=features,
    )


def dataset_dict_cast_column(
    input_dataset_dict: DatasetDict, column: str, feature
) -> "DatasetDict":
    return input_dataset_dict.cast_column(
        column=column,
        feature=feature,
    )


def dataset_dict_class_encode_column(
    input_dataset_dict: DatasetDict, column: str, include_nulls: bool = False
) -> "DatasetDict":
    return input_dataset_dict.class_encode_column(
        column=column,
        include_nulls=include_nulls,
    )


def dataset_dict_cleanup_cache_files(input_dataset_dict: DatasetDict) -> Dict[str, int]:
    return input_dataset_dict.cleanup_cache_files()


def dataset_dict_filter(
    input_dataset_dict: DatasetDict,
    function,
    with_indices=False,
    input_columns: Union[str, List[str], None] = None,
    batched: bool = False,
    batch_size: Optional[int] = 1000,
    keep_in_memory: bool = False,
    load_from_cache_file: Optional[bool] = None,
    cache_file_names: Optional[Dict[str, Optional[str]]] = None,
    writer_batch_size: Optional[int] = 1000,
    fn_kwargs: Optional[dict] = None,
    num_proc: Optional[int] = None,
    desc: Optional[str] = None,
) -> "DatasetDict":
    return input_dataset_dict.filter(
        function=function,
        with_indices=with_indices,
        input_columns=input_columns,
        batched=batched,
        batch_size=batch_size,
        keep_in_memory=keep_in_memory,
        load_from_cache_file=load_from_cache_file,
        cache_file_names=cache_file_names,
        writer_batch_size=writer_batch_size,
        fn_kwargs=fn_kwargs,
        num_proc=num_proc,
        desc=desc,
    )


def dataset_dict_flatten(
    input_dataset_dict: DatasetDict, max_depth=16
) -> "DatasetDict":
    return input_dataset_dict.flatten(
        max_depth=max_depth,
    )


def dataset_dict_flatten_indices(
    input_dataset_dict: DatasetDict,
    keep_in_memory: bool = False,
    cache_file_names: Optional[Dict[str, Optional[str]]] = None,
    writer_batch_size: Optional[int] = 1000,
    features: Optional[datasets.features.features.Features] = None,
    disable_nullable: bool = False,
    num_proc: Optional[int] = None,
    new_fingerprint: Optional[str] = None,
) -> "DatasetDict":
    return input_dataset_dict.flatten_indices(
        keep_in_memory=keep_in_memory,
        cache_file_names=cache_file_names,
        writer_batch_size=writer_batch_size,
        features=features,
        disable_nullable=disable_nullable,
        num_proc=num_proc,
        new_fingerprint=new_fingerprint,
    )


def dataset_dict_formatted_as(
    input_dataset_dict: DatasetDict,
    type: Optional[str] = None,
    columns: Optional[List] = None,
    output_all_columns: bool = False,
    **format_kwargs,
) -> None:
    return input_dataset_dict.formatted_as(
        type=type,
        columns=columns,
        output_all_columns=output_all_columns,
        format_kwargs=format_kwargs,
    )


def dataset_dict_from_csv(
    input_dataset_dict: DatasetDict,
    path_or_paths: Dict[str, Union[str, bytes, os.PathLike]],
    features: Optional[datasets.features.features.Features] = None,
    cache_dir: str = None,
    keep_in_memory: bool = False,
    **kwargs,
):
    return input_dataset_dict.from_csv(
        path_or_paths=path_or_paths,
        features=features,
        cache_dir=cache_dir,
        keep_in_memory=keep_in_memory,
        kwargs=kwargs,
    )


def dataset_dict_from_json(
    input_dataset_dict: DatasetDict,
    path_or_paths: Dict[str, Union[str, bytes, os.PathLike]],
    features: Optional[datasets.features.features.Features] = None,
    cache_dir: str = None,
    keep_in_memory: bool = False,
    **kwargs,
):
    return input_dataset_dict.from_json(
        path_or_paths=path_or_paths,
        features=features,
        cache_dir=cache_dir,
        keep_in_memory=keep_in_memory,
        kwargs=kwargs,
    )


def dataset_dict_from_parquet(
    input_dataset_dict: DatasetDict,
    path_or_paths: Dict[str, Union[str, bytes, os.PathLike]],
    features: Optional[datasets.features.features.Features] = None,
    cache_dir: str = None,
    keep_in_memory: bool = False,
    columns: Optional[List[str]] = None,
    **kwargs,
):
    return input_dataset_dict.from_parquet(
        path_or_paths=path_or_paths,
        features=features,
        cache_dir=cache_dir,
        keep_in_memory=keep_in_memory,
        columns=columns,
        kwargs=kwargs,
    )


def dataset_dict_from_text(
    input_dataset_dict: DatasetDict,
    path_or_paths: Dict[str, Union[str, bytes, os.PathLike]],
    features: Optional[datasets.features.features.Features] = None,
    cache_dir: str = None,
    keep_in_memory: bool = False,
    **kwargs,
):
    return input_dataset_dict.from_text(
        path_or_paths=path_or_paths,
        features=features,
        cache_dir=cache_dir,
        keep_in_memory=keep_in_memory,
        kwargs=kwargs,
    )


def dataset_dict_load_from_disk(
    input_dataset_dict: DatasetDict,
    dataset_dict_path: Union[str, bytes, os.PathLike],
    fs,
    keep_in_memory: Optional[bool] = None,
    storage_options: Optional[dict] = None,
):
    return input_dataset_dict.load_from_disk(
        dataset_dict_path=dataset_dict_path,
        fs=fs,
        keep_in_memory=keep_in_memory,
        storage_options=storage_options,
    )


def dataset_dict_map(
    input_dataset_dict: DatasetDict,
    function: Optional[Callable] = None,
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
    features: Optional[datasets.features.features.Features] = None,
    disable_nullable: bool = False,
    fn_kwargs: Optional[dict] = None,
    num_proc: Optional[int] = None,
    desc: Optional[str] = None,
) -> "DatasetDict":
    return input_dataset_dict.map(
        function=function,
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


def dataset_dict_prepare_for_task(
    input_dataset_dict: DatasetDict,
    task: Union[str, datasets.tasks.base.TaskTemplate],
    id: int = 0,
) -> "DatasetDict":
    return input_dataset_dict.prepare_for_task(
        task=task,
        id=id,
    )


def dataset_dict_push_to_hub(
    input_dataset_dict: DatasetDict,
    repo_id,
    config_name: str = "default",
    set_default: Optional[bool] = None,
    commit_message: Optional[str] = None,
    commit_description: Optional[str] = None,
    private: Optional[bool] = False,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    branch="deprecated",
    create_pr: Optional[bool] = False,
    max_shard_size: Union[str, int, None] = None,
    num_shards: Optional[Dict[str, int]] = None,
    embed_external_files: bool = True,
) -> huggingface_hub.hf_api.CommitInfo:
    return input_dataset_dict.push_to_hub(
        repo_id=repo_id,
        config_name=config_name,
        set_default=set_default,
        commit_message=commit_message,
        commit_description=commit_description,
        private=private,
        token=token,
        revision=revision,
        branch=branch,
        create_pr=create_pr,
        max_shard_size=max_shard_size,
        num_shards=num_shards,
        embed_external_files=embed_external_files,
    )


def dataset_dict_remove_columns(
    input_dataset_dict: DatasetDict, column_names: Union[str, List[str]]
) -> "DatasetDict":
    return input_dataset_dict.remove_columns(
        column_names=column_names,
    )


def dataset_dict_rename_column(
    input_dataset_dict: DatasetDict, original_column_name: str, new_column_name: str
) -> "DatasetDict":
    return input_dataset_dict.rename_column(
        original_column_name=original_column_name,
        new_column_name=new_column_name,
    )


def dataset_dict_rename_columns(
    input_dataset_dict: DatasetDict, column_mapping: Dict[str, str]
) -> "DatasetDict":
    return input_dataset_dict.rename_columns(
        column_mapping=column_mapping,
    )


def dataset_dict_reset_format(input_dataset_dict: DatasetDict) -> None:
    return input_dataset_dict.reset_format()


def dataset_dict_save_to_disk(
    input_dataset_dict: DatasetDict,
    dataset_dict_path: Union[str, bytes, os.PathLike],
    fs="deprecated",
    max_shard_size: Union[str, int, None] = None,
    num_shards: Optional[Dict[str, int]] = None,
    num_proc: Optional[int] = None,
    storage_options: Optional[dict] = None,
) -> None:
    return input_dataset_dict.save_to_disk(
        dataset_dict_path=dataset_dict_path,
        fs=fs,
        max_shard_size=max_shard_size,
        num_shards=num_shards,
        num_proc=num_proc,
        storage_options=storage_options,
    )


def dataset_dict_select_columns(
    input_dataset_dict: DatasetDict, column_names: Union[str, List[str]]
) -> "DatasetDict":
    return input_dataset_dict.select_columns(
        column_names=column_names,
    )


def dataset_dict_set_format(
    input_dataset_dict: DatasetDict,
    type: Optional[str] = None,
    columns: Optional[List] = None,
    output_all_columns: bool = False,
    **format_kwargs,
) -> None:
    return input_dataset_dict.set_format(
        type=type,
        columns=columns,
        output_all_columns=output_all_columns,
        format_kwargs=format_kwargs,
    )


def dataset_dict_set_transform(
    input_dataset_dict: DatasetDict,
    transform: Optional[Callable],
    columns: Optional[List] = None,
    output_all_columns: bool = False,
) -> None:
    return input_dataset_dict.set_transform(
        transform=transform,
        columns=columns,
        output_all_columns=output_all_columns,
    )


def dataset_dict_shuffle(
    input_dataset_dict: DatasetDict,
    seeds: Union[int, Dict[str, Optional[int]], None] = None,
    seed: Optional[int] = None,
    generators: Optional[Dict[str, np.random._generator.Generator]] = None,
    keep_in_memory: bool = False,
    load_from_cache_file: Optional[bool] = None,
    indices_cache_file_names: Optional[Dict[str, Optional[str]]] = None,
    writer_batch_size: Optional[int] = 1000,
) -> "DatasetDict":
    return input_dataset_dict.shuffle(
        seeds=seeds,
        seed=seed,
        generators=generators,
        keep_in_memory=keep_in_memory,
        load_from_cache_file=load_from_cache_file,
        indices_cache_file_names=indices_cache_file_names,
        writer_batch_size=writer_batch_size,
    )


def dataset_dict_sort(
    input_dataset_dict: DatasetDict,
    column_names: Union[str, Sequence[str]],
    reverse: Union[bool, Sequence[bool]] = False,
    kind="deprecated",
    null_placement: str = "at_end",
    keep_in_memory: bool = False,
    load_from_cache_file: Optional[bool] = None,
    indices_cache_file_names: Optional[Dict[str, Optional[str]]] = None,
    writer_batch_size: Optional[int] = 1000,
) -> "DatasetDict":
    return input_dataset_dict.sort(
        column_names=column_names,
        reverse=reverse,
        kind=kind,
        null_placement=null_placement,
        keep_in_memory=keep_in_memory,
        load_from_cache_file=load_from_cache_file,
        indices_cache_file_names=indices_cache_file_names,
        writer_batch_size=writer_batch_size,
    )


def dataset_dict_unique(
    input_dataset_dict: DatasetDict, column: str
) -> Dict[str, List]:
    return input_dataset_dict.unique(
        column=column,
    )


def dataset_dict_with_format(
    input_dataset_dict: DatasetDict,
    type: Optional[str] = None,
    columns: Optional[List] = None,
    output_all_columns: bool = False,
    **format_kwargs,
) -> "DatasetDict":
    return input_dataset_dict.with_format(
        type=type,
        columns=columns,
        output_all_columns=output_all_columns,
        format_kwargs=format_kwargs,
    )


def dataset_dict_with_transform(
    input_dataset_dict: DatasetDict,
    transform: Optional[Callable],
    columns: Optional[List] = None,
    output_all_columns: bool = False,
) -> "DatasetDict":
    return input_dataset_dict.with_transform(
        transform=transform,
        columns=columns,
        output_all_columns=output_all_columns,
    )
