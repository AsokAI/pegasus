import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import datasets
import huggingface_hub
import numpy
from datasets import DatasetDict


def enter(dataset: DatasetDict) -> None:
	return dataset.__enter__()


def exit(dataset: DatasetDict, exc_type, exc_val, exc_tb) -> None:
	return dataset.__exit__(exc_type, exc_val, exc_tb)


def getitem(dataset: DatasetDict, k) -> datasets.arrow_dataset.Dataset:
	return dataset.__getitem__(k)


def repr(dataset: DatasetDict) -> None:
	return dataset.__repr__()


def checkvaluesfeatures(dataset: DatasetDict) -> None:
	return dataset._check_values_features()


def checkvaluestype(dataset: DatasetDict) -> None:
	return dataset._check_values_type()


def align_labels_with_mapping(
    dataset: DatasetDict, label2id: Dict, label_column: str
) -> "DatasetDict":
    return dataset.align_labels_with_mapping(label2id, label_column)


def cast(
    dataset: DatasetDict, features: datasets.features.features.Features
) -> "DatasetDict":
    return dataset.cast(features)


def cast_column(dataset: DatasetDict, column: str, feature) -> "DatasetDict":
    return dataset.cast_column(column, feature)


def class_encode_column(
    dataset: DatasetDict, column: str, include_nulls: bool = False
) -> "DatasetDict":
    return dataset.class_encode_column(column, include_nulls)


def cleanup_cache_files(dataset: DatasetDict) -> Dict[str, int]:
    return dataset.cleanup_cache_files()


def filter(
    dataset: DatasetDict,
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
    return dataset.filter(
        function,
        with_indices,
        input_columns,
        batched,
        batch_size,
        keep_in_memory,
        load_from_cache_file,
        cache_file_names,
        writer_batch_size,
        fn_kwargs,
        num_proc,
        desc,
    )


def flatten(dataset: DatasetDict, max_depth=16) -> "DatasetDict":
    return dataset.flatten(max_depth)


def flatten_indices(
    dataset: DatasetDict,
    keep_in_memory: bool = False,
    cache_file_names: Optional[Dict[str, Optional[str]]] = None,
    writer_batch_size: Optional[int] = 1000,
    features: Optional[datasets.features.features.Features] = None,
    disable_nullable: bool = False,
    num_proc: Optional[int] = None,
    new_fingerprint: Optional[str] = None,
) -> "DatasetDict":
    return dataset.flatten_indices(
        keep_in_memory,
        cache_file_names,
        writer_batch_size,
        features,
        disable_nullable,
        num_proc,
        new_fingerprint,
    )


def formatted_as(
    dataset: DatasetDict,
    type: Optional[str] = None,
    columns: Optional[List] = None,
    output_all_columns: bool = False,
    **format_kwargs
) -> None:
    return dataset.formatted_as(type, columns, output_all_columns, format_kwargs)


def from_csv(
    dataset: DatasetDict,
    path_or_paths: Dict,
    features: Optional[datasets.Features],
    cache_dir: str,
    keep_in_memory: bool,
    kwargs,
):
    return dataset.from_csv(path_or_paths, features, cache_dir, keep_in_memory, kwargs)


def from_json(
    dataset: DatasetDict,
    path_or_paths: Dict,
    features: Optional[datasets.Features],
    cache_dir: str,
    keep_in_memory: bool,
    kwargs,
):
    return dataset.from_json(path_or_paths, features, cache_dir, keep_in_memory, kwargs)


def from_parquet(
    dataset: DatasetDict,
    path_or_paths: Dict,
    features: Optional[datasets.Features],
    cache_dir: str,
    keep_in_memory: bool,
    columns: Optional[List[str]],
    kwargs,
):
    return dataset.from_parquet(
        path_or_paths, features, cache_dir, keep_in_memory, columns, kwargs
    )


def from_text(
    dataset: DatasetDict,
    path_or_paths: Dict,
    features: Optional[datasets.Features],
    cache_dir: str,
    keep_in_memory: bool,
    kwargs,
):
    return dataset.from_text(path_or_paths, features, cache_dir, keep_in_memory, kwargs)


def load_from_disk(
    dataset: DatasetDict,
    dataset_dict_path: Union[str, bytes, os.PathLike],
    fs,
    keep_in_memory: Optional[bool],
    storage_options: Optional[Dict],
):
    return dataset.load_from_disk(
        dataset_dict_path, fs, keep_in_memory, storage_options
    )


def map(
    dataset: DatasetDict,
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
    return dataset.map(
        function,
        with_indices,
        with_rank,
        input_columns,
        batched,
        batch_size,
        drop_last_batch,
        remove_columns,
        keep_in_memory,
        load_from_cache_file,
        cache_file_names,
        writer_batch_size,
        features,
        disable_nullable,
        fn_kwargs,
        num_proc,
        desc,
    )


def prepare_for_task(
    dataset: DatasetDict,
    task: Union[str, datasets.tasks.base.TaskTemplate],
    id: int = 0,
) -> "DatasetDict":
    return dataset.prepare_for_task(task, id)


def push_to_hub(
    dataset: DatasetDict,
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
    return dataset.push_to_hub(
        repo_id,
        config_name,
        set_default,
        commit_message,
        commit_description,
        private,
        token,
        revision,
        branch,
        create_pr,
        max_shard_size,
        num_shards,
        embed_external_files,
    )


def remove_columns(
    dataset: DatasetDict, column_names: Union[str, List[str]]
) -> "DatasetDict":
    return dataset.remove_columns(column_names)


def rename_column(
    dataset: DatasetDict, original_column_name: str, new_column_name: str
) -> "DatasetDict":
    return dataset.rename_column(original_column_name, new_column_name)


def rename_columns(
    dataset: DatasetDict, column_mapping: Dict[str, str]
) -> "DatasetDict":
    return dataset.rename_columns(column_mapping)


def reset_format(dataset: DatasetDict) -> None:
    return dataset.reset_format()


def save_to_disk(
    dataset: DatasetDict,
    dataset_dict_path: Union[str, bytes, os.PathLike],
    fs="deprecated",
    max_shard_size: Union[str, int, None] = None,
    num_shards: Optional[Dict[str, int]] = None,
    num_proc: Optional[int] = None,
    storage_options: Optional[dict] = None,
) -> None:
    return dataset.save_to_disk(
        dataset_dict_path, fs, max_shard_size, num_shards, num_proc, storage_options
    )


def select_columns(
    dataset: DatasetDict, column_names: Union[str, List[str]]
) -> "DatasetDict":
    return dataset.select_columns(column_names)


def set_format(
    dataset: DatasetDict,
    type: Optional[str] = None,
    columns: Optional[List] = None,
    output_all_columns: bool = False,
    **format_kwargs
) -> None:
    return dataset.set_format(type, columns, output_all_columns, format_kwargs)


def set_transform(
    dataset: DatasetDict,
    transform: Optional[Callable],
    columns: Optional[List] = None,
    output_all_columns: bool = False,
) -> None:
    return dataset.set_transform(transform, columns, output_all_columns)


def shuffle(
    dataset: DatasetDict,
    seeds: Union[int, Dict[str, Optional[int]], None] = None,
    seed: Optional[int] = None,
    generators: Optional[Dict[str, numpy.random._generator.Generator]] = None,
    keep_in_memory: bool = False,
    load_from_cache_file: Optional[bool] = None,
    indices_cache_file_names: Optional[Dict[str, Optional[str]]] = None,
    writer_batch_size: Optional[int] = 1000,
) -> "DatasetDict":
    return dataset.shuffle(
        seeds,
        seed,
        generators,
        keep_in_memory,
        load_from_cache_file,
        indices_cache_file_names,
        writer_batch_size,
    )


def sort(
    dataset: DatasetDict,
    column_names: Union[str, Sequence[str]],
    reverse: Union[bool, Sequence[bool]] = False,
    kind="deprecated",
    null_placement: str = "at_end",
    keep_in_memory: bool = False,
    load_from_cache_file: Optional[bool] = None,
    indices_cache_file_names: Optional[Dict[str, Optional[str]]] = None,
    writer_batch_size: Optional[int] = 1000,
) -> "DatasetDict":
    return dataset.sort(
        column_names,
        reverse,
        kind,
        null_placement,
        keep_in_memory,
        load_from_cache_file,
        indices_cache_file_names,
        writer_batch_size,
    )


def unique(dataset: DatasetDict, column: str) -> Dict[str, List]:
    return dataset.unique(column)


def with_format(
    dataset: DatasetDict,
    type: Optional[str] = None,
    columns: Optional[List] = None,
    output_all_columns: bool = False,
    **format_kwargs
) -> "DatasetDict":
    return dataset.with_format(type, columns, output_all_columns, format_kwargs)


def with_transform(
    dataset: DatasetDict,
    transform: Optional[Callable],
    columns: Optional[List] = None,
    output_all_columns: bool = False,
) -> "DatasetDict":
    return dataset.with_transform(transform, columns, output_all_columns)
