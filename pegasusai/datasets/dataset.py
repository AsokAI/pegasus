import os
import pathlib
from typing import Iterable, List, Optional, Union, Tuple, Dict, Any, Callable

import datasets
import numpy as np
import pyarrow
from datasets import Dataset, Features


def __del__(input_dataset: Dataset) -> None:
    return input_dataset.__del__()


def __enter__(input_dataset: Dataset) -> None:
    return input_dataset.__enter__()


def __exit__(input_dataset: Dataset, exc_type, exc_val, exc_tb) -> None:
    return input_dataset.__exit__(exc_type, exc_val, exc_tb)


def __getitem__(input_dataset: Dataset, key) -> None:
    return input_dataset.__getitem__(key)


def __getitems__(input_dataset: Dataset, keys: List) -> List:
    return input_dataset.__getitems__(keys)


def __init__(
    input_dataset: Dataset,
    arrow_table: datasets.table.Table,
    info: Optional[datasets.info.DatasetInfo] = None,
    split: Optional[datasets.splits.NamedSplit] = None,
    indices_table: Optional[datasets.table.Table] = None,
    fingerprint: Optional[str] = None,
) -> None:
    return input_dataset.__init__(arrow_table, info, split, indices_table, fingerprint)


def __iter__(input_dataset: Dataset) -> None:
    return input_dataset.__iter__()


def __len__(input_dataset: Dataset) -> None:
    return input_dataset.__len__()


def __repr__(input_dataset: Dataset) -> None:
    return input_dataset.__repr__()


def __setstate__(input_dataset: Dataset, state) -> None:
    return input_dataset.__setstate__(state)


def _build_local_temp_path(
    input_dataset: Dataset,
    uri_or_path: str,
):
    return input_dataset._build_local_temp_path(uri_or_path)


def _check_index_is_initialized(input_dataset: Dataset, index_name: str) -> None:
    return input_dataset._check_index_is_initialized(index_name)


def _estimate_nbytes(input_dataset: Dataset) -> int:
    return input_dataset._estimate_nbytes()


def _generate_tables_from_cache_file(
    input_dataset: Dataset,
    filename: str,
) -> None:
    return input_dataset._generate_tables_from_cache_file(filename)


def _generate_tables_from_shards(
    input_dataset: Dataset,
    shards: List[Dataset],
    batch_size: int,
) -> None:
    return input_dataset._generate_tables_from_shards(shards, batch_size)


def _get_cache_file_path(input_dataset: Dataset, fingerprint) -> None:
    return input_dataset._get_cache_file_path(fingerprint)


def _get_output_signature(
    input_dataset: Dataset,
    dataset: "Dataset",
    collate_fn: Callable,
    collate_fn_args: dict,
    cols_to_retain: Optional[List[str]] = None,
    batch_size: Optional[int] = None,
    num_test_batches: int = 20,
) -> None:
    return input_dataset._get_output_signature(
        dataset,
        collate_fn,
        collate_fn_args,
        cols_to_retain,
        batch_size,
        num_test_batches,
    )


def _getitem(
    input_dataset: Dataset,
    key: Union[int, slice, str, List[int], Tuple[int, ...]],
    **kwargs,
) -> Union[Dict, List]:
    return input_dataset._getitem(key, kwargs)


def _map_single(
    input_dataset: Dataset,
    shard: "Dataset",
    function: Optional[Callable] = None,
    with_indices: bool = False,
    with_rank: bool = False,
    input_columns: Optional[List[str]] = None,
    batched: bool = False,
    batch_size: Optional[int] = 1000,
    drop_last_batch: bool = False,
    remove_columns: Optional[List[str]] = None,
    keep_in_memory: bool = False,
    cache_file_name: Optional[str] = None,
    writer_batch_size: Optional[int] = 1000,
    features: Optional[datasets.features.features.Features] = None,
    disable_nullable: bool = False,
    fn_kwargs: Optional[dict] = None,
    new_fingerprint: Optional[str] = None,
    rank: Optional[int] = None,
    offset: int = 0,
):
    return input_dataset._map_single(
        shard,
        function,
        with_indices,
        with_rank,
        input_columns,
        batched,
        batch_size,
        drop_last_batch,
        remove_columns,
        keep_in_memory,
        cache_file_name,
        writer_batch_size,
        features,
        disable_nullable,
        fn_kwargs,
        new_fingerprint,
        rank,
        offset,
    )


def _new_dataset_with_indices(
    input_dataset: Dataset,
    indices_cache_file_name: Optional[str] = None,
    indices_buffer: Optional[pyarrow.lib.Buffer] = None,
    fingerprint: Optional[str] = None,
) -> "Dataset":
    return input_dataset._new_dataset_with_indices(
        indices_cache_file_name, indices_buffer, fingerprint
    )


def _push_parquet_shards_to_hub(
    input_dataset: Dataset,
    repo_id: str,
    data_dir: str = "data",
    split: Optional[str] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    create_pr: Optional[bool] = False,
    max_shard_size: Union[str, int, None] = None,
    num_shards: Optional[int] = None,
    embed_external_files: bool = True,
) -> Tuple[str, str, int, int, List[str], int]:
    return input_dataset._push_parquet_shards_to_hub(
        repo_id,
        data_dir,
        split,
        token,
        revision,
        create_pr,
        max_shard_size,
        num_shards,
        embed_external_files,
    )


def _save_to_disk_single(
    input_dataset: Dataset,
    job_id: int,
    shard: "Dataset",
    fpath: str,
    storage_options: Optional[dict],
) -> None:
    return input_dataset._save_to_disk_single(job_id, shard, fpath, storage_options)


def _select_contiguous(
    input_dataset: Dataset,
    start: int,
    length: int,
    new_fingerprint: Optional[str] = None,
) -> "Dataset":
    return input_dataset._select_contiguous(start, length, new_fingerprint)


def _select_with_indices_mapping(
    input_dataset: Dataset,
    indices: Iterable,
    keep_in_memory: bool = False,
    indices_cache_file_name: Optional[str] = None,
    writer_batch_size: Optional[int] = 1000,
    new_fingerprint: Optional[str] = None,
) -> "Dataset":
    return input_dataset._select_with_indices_mapping(
        indices,
        keep_in_memory,
        indices_cache_file_name,
        writer_batch_size,
        new_fingerprint,
    )


def add_column(
    input_dataset: Dataset,
    name: str,
    column: Union[list, np.array],
    new_fingerprint: str,
) -> None:
    return input_dataset.add_column(name, column, new_fingerprint)


def add_elasticsearch_index(
    input_dataset: Dataset,
    column: str,
    index_name: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    es_client: Optional["elasticsearch.Elasticsearch"] = None,
    es_index_name: Optional[str] = None,
    es_index_config: Optional[dict] = None,
) -> None:
    return input_dataset.add_elasticsearch_index(
        column, index_name, host, port, es_client, es_index_name, es_index_config
    )


def add_faiss_index(
    input_dataset: Dataset,
    column: str,
    index_name: Optional[str] = None,
    device: Optional[int] = None,
    string_factory: Optional[str] = None,
    metric_type: Optional[int] = None,
    custom_index: Optional["faiss.Index"] = None,
    batch_size: int = 1000,
    train_size: Optional[int] = None,
    faiss_verbose: bool = False,
    dtype=np.float32,
) -> None:
    return input_dataset.add_faiss_index(
        column,
        index_name,
        device,
        string_factory,
        metric_type,
        custom_index,
        batch_size,
        train_size,
        faiss_verbose,
        dtype,
    )


def add_faiss_index_from_external_arrays(
    input_dataset: Dataset,
    external_arrays: np.array,
    index_name: str,
    device: Optional[int] = None,
    string_factory: Optional[str] = None,
    metric_type: Optional[int] = None,
    custom_index: Optional["faiss.Index"] = None,
    batch_size: int = 1000,
    train_size: Optional[int] = None,
    faiss_verbose: bool = False,
    dtype=np.float32,
) -> None:
    return input_dataset.add_faiss_index_from_external_arrays(
        external_arrays,
        index_name,
        device,
        string_factory,
        metric_type,
        custom_index,
        batch_size,
        train_size,
        faiss_verbose,
        dtype,
    )


def add_item(input_dataset: Dataset, item: dict, new_fingerprint: str) -> None:
    return input_dataset.add_item(item, new_fingerprint)


def align_labels_with_mapping(
    input_dataset: Dataset, label2id: Dict, label_column: str
) -> "Dataset":
    return input_dataset.align_labels_with_mapping(label2id, label_column)


def cast(
    input_dataset: Dataset,
    features: datasets.features.features.Features,
    batch_size: Optional[int] = 1000,
    keep_in_memory: bool = False,
    load_from_cache_file: Optional[bool] = None,
    cache_file_name: Optional[str] = None,
    writer_batch_size: Optional[int] = 1000,
    num_proc: Optional[int] = None,
) -> "Dataset":
    return input_dataset.cast(
        features,
        batch_size,
        keep_in_memory,
        load_from_cache_file,
        cache_file_name,
        writer_batch_size,
        num_proc,
    )


def cast_column(
    input_dataset: Dataset,
    column: str,
    feature: Union[
        dict,
        list,
        tuple,
        datasets.features.features.Value,
        datasets.features.features.ClassLabel,
        datasets.features.translation.Translation,
        datasets.features.translation.TranslationVariableLanguages,
        datasets.features.features.Sequence,
        datasets.features.features.Array2D,
        datasets.features.features.Array3D,
        datasets.features.features.Array4D,
        datasets.features.features.Array5D,
        datasets.features.audio.Audio,
        datasets.features.image.Image,
    ],
    new_fingerprint: Optional[str] = None,
) -> "Dataset":
    return input_dataset.cast_column(column, feature, new_fingerprint)


def class_encode_column(
    input_dataset: Dataset, column: str, include_nulls: bool = False
) -> "Dataset":
    return input_dataset.class_encode_column(column, include_nulls)


def cleanup_cache_files(input_dataset: Dataset) -> int:
    return input_dataset.cleanup_cache_files()


def drop_index(input_dataset: Dataset, index_name: str) -> None:
    return input_dataset.drop_index(index_name)


def export(input_dataset: Dataset, filename: str, format: str = "tfrecord") -> None:
    return input_dataset.export(filename, format)


def filter(
    input_dataset: Dataset,
    function: Optional[Callable] = None,
    with_indices=False,
    input_columns: Union[str, List[str], None] = None,
    batched: bool = False,
    batch_size: Optional[int] = 1000,
    keep_in_memory: bool = False,
    load_from_cache_file: Optional[bool] = None,
    cache_file_name: Optional[str] = None,
    writer_batch_size: Optional[int] = 1000,
    fn_kwargs: Optional[dict] = None,
    num_proc: Optional[int] = None,
    suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
    new_fingerprint: Optional[str] = None,
    desc: Optional[str] = None,
) -> "Dataset":
    return input_dataset.filter(
        function,
        with_indices,
        input_columns,
        batched,
        batch_size,
        keep_in_memory,
        load_from_cache_file,
        cache_file_name,
        writer_batch_size,
        fn_kwargs,
        num_proc,
        suffix_template,
        new_fingerprint,
        desc,
    )


def flatten(
    input_dataset: Dataset, new_fingerprint: Optional[str] = None, max_depth=16
) -> "Dataset":
    return input_dataset.flatten(new_fingerprint, max_depth)


def flatten_indices(
    input_dataset: Dataset,
    keep_in_memory: bool = False,
    cache_file_name: Optional[str] = None,
    writer_batch_size: Optional[int] = 1000,
    features: Optional[datasets.features.features.Features] = None,
    disable_nullable: bool = False,
    num_proc: Optional[int] = None,
    new_fingerprint: Optional[str] = None,
) -> "Dataset":
    return input_dataset.flatten_indices(
        keep_in_memory,
        cache_file_name,
        writer_batch_size,
        features,
        disable_nullable,
        num_proc,
        new_fingerprint,
    )


def formatted_as(
    input_dataset: Dataset,
    type: Optional[str] = None,
    columns: Optional[List] = None,
    output_all_columns: bool = False,
    **format_kwargs,
) -> None:
    return input_dataset.formatted_as(type, columns, output_all_columns, format_kwargs)


def from_csv(
    input_dataset: Dataset,
    path_or_paths: Union[str, bytes, os.PathLike, List[Union[str, bytes, os.PathLike]]],
    split: Optional[datasets.splits.NamedSplit] = None,
    features: Optional[datasets.features.features.Features] = None,
    cache_dir: str = None,
    keep_in_memory: bool = False,
    num_proc: Optional[int] = None,
    **kwargs,
) -> None:
    return input_dataset.from_csv(
        path_or_paths, split, features, cache_dir, keep_in_memory, num_proc, kwargs
    )


def from_generator(
    input_dataset: Dataset,
    generator: Callable,
    features: Optional[datasets.features.features.Features] = None,
    cache_dir: str = None,
    keep_in_memory: bool = False,
    gen_kwargs: Optional[dict] = None,
    num_proc: Optional[int] = None,
    **kwargs,
) -> None:
    return input_dataset.from_generator(
        generator, features, cache_dir, keep_in_memory, gen_kwargs, num_proc, kwargs
    )


def from_json(
    input_dataset: Dataset,
    path_or_paths: Union[str, bytes, os.PathLike, List[Union[str, bytes, os.PathLike]]],
    split: Optional[datasets.splits.NamedSplit] = None,
    features: Optional[datasets.features.features.Features] = None,
    cache_dir: str = None,
    keep_in_memory: bool = False,
    field: Optional[str] = None,
    num_proc: Optional[int] = None,
    **kwargs,
) -> None:
    return input_dataset.from_json(
        path_or_paths,
        split,
        features,
        cache_dir,
        keep_in_memory,
        field,
        num_proc,
        kwargs,
    )


def from_parquet(
    input_dataset: Dataset,
    path_or_paths: Union[str, bytes, os.PathLike, List[Union[str, bytes, os.PathLike]]],
    split: Optional[datasets.splits.NamedSplit] = None,
    features: Optional[datasets.features.features.Features] = None,
    cache_dir: str = None,
    keep_in_memory: bool = False,
    columns: Optional[List[str]] = None,
    num_proc: Optional[int] = None,
    **kwargs,
) -> None:
    return input_dataset.from_parquet(
        path_or_paths,
        split,
        features,
        cache_dir,
        keep_in_memory,
        columns,
        num_proc,
        kwargs,
    )


def from_spark(
    input_dataset: Dataset,
    df: "pyspark.sql.DataFrame",
    split: Optional[datasets.splits.NamedSplit] = None,
    features: Optional[datasets.features.features.Features] = None,
    keep_in_memory: bool = False,
    cache_dir: str = None,
    working_dir: str = None,
    load_from_cache_file: bool = True,
    **kwargs,
) -> None:
    return input_dataset.from_spark(
        df,
        split,
        features,
        keep_in_memory,
        cache_dir,
        working_dir,
        load_from_cache_file,
        kwargs,
    )


def from_sql(
    input_dataset: Dataset,
    sql: Union[str, "sqlalchemy.sql.Selectable"],
    con: Union[
        str,
        "sqlalchemy.engine.Connection",
        "sqlalchemy.engine.Engine",
        "sqlite3.Connection",
    ],
    features: Optional[datasets.features.features.Features] = None,
    cache_dir: str = None,
    keep_in_memory: bool = False,
    **kwargs,
) -> None:
    return input_dataset.from_sql(sql, con, features, cache_dir, keep_in_memory, kwargs)


def from_text(
    input_dataset: Dataset,
    path_or_paths: Union[str, bytes, os.PathLike, List[Union[str, bytes, os.PathLike]]],
    split: Optional[datasets.splits.NamedSplit] = None,
    features: Optional[datasets.features.features.Features] = None,
    cache_dir: str = None,
    keep_in_memory: bool = False,
    num_proc: Optional[int] = None,
    **kwargs,
) -> None:
    return input_dataset.from_text(
        path_or_paths, split, features, cache_dir, keep_in_memory, num_proc, kwargs
    )


def get_index(input_dataset: Dataset, index_name: str) -> datasets.search.BaseIndex:
    return input_dataset.get_index(index_name)


def get_nearest_examples(
    input_dataset: Dataset,
    index_name: str,
    query: Union[str, np.array],
    k: int = 10,
    **kwargs,
) -> datasets.search.NearestExamplesResults:
    return input_dataset.get_nearest_examples(index_name, query, k, kwargs)


def get_nearest_examples_batch(
    input_dataset: Dataset,
    index_name: str,
    queries: Union[List[str], np.array],
    k: int = 10,
    **kwargs,
) -> datasets.search.BatchedNearestExamplesResults:
    return input_dataset.get_nearest_examples_batch(index_name, queries, k, kwargs)


def is_index_initialized(input_dataset: Dataset, index_name: str) -> bool:
    return input_dataset.is_index_initialized(index_name)


def iter(
    input_dataset: Dataset, batch_size: int, drop_last_batch: bool = False
) -> None:
    return input_dataset.iter(batch_size, drop_last_batch)


def list_indexes(input_dataset: Dataset) -> List[str]:
    return input_dataset.list_indexes()


def load_elasticsearch_index(
    input_dataset: Dataset,
    index_name: str,
    es_index_name: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    es_client: Optional[ForwardRef("Elasticsearch")] = None,
    es_index_config: Optional[dict] = None,
) -> None:
    return input_dataset.load_elasticsearch_index(
        index_name, es_index_name, host, port, es_client, es_index_config
    )


def load_faiss_index(
    input_dataset: Dataset,
    index_name: str,
    file: Union[str, pathlib.PurePath],
    device: Union[int, List[int], None] = None,
    storage_options: Optional[Dict] = None,
) -> None:
    return input_dataset.load_faiss_index(index_name, file, device, storage_options)


def load_from_disk(
    input_dataset: Dataset,
    dataset_path: str,
    fs,
    keep_in_memory: Optional[bool] = None,
    storage_options: Optional[dict] = None,
):
    return input_dataset.load_from_disk(
        dataset_path, fs, keep_in_memory, storage_options
    )


def map(
    input_dataset: Dataset,
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
    cache_file_name: Optional[str] = None,
    writer_batch_size: Optional[int] = 1000,
    features: Optional[datasets.features.features.Features] = None,
    disable_nullable: bool = False,
    fn_kwargs: Optional[dict] = None,
    num_proc: Optional[int] = None,
    suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
    new_fingerprint: Optional[str] = None,
    desc: Optional[str] = None,
) -> "Dataset":
    return input_dataset.map(
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
        cache_file_name,
        writer_batch_size,
        features,
        disable_nullable,
        fn_kwargs,
        num_proc,
        suffix_template,
        new_fingerprint,
        desc,
    )


def prepare_for_task(
    input_dataset: Dataset,
    task: Union[str, datasets.tasks.base.TaskTemplate],
    id: int = 0,
) -> "Dataset":
    return input_dataset.prepare_for_task(task, id)


def push_to_hub(
    input_dataset: Dataset,
    repo_id: str,
    config_name: str = "default",
    set_default: Optional[bool] = None,
    split: Optional[str] = None,
    commit_message: Optional[str] = None,
    commit_description: Optional[str] = None,
    private: Optional[bool] = False,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    branch="deprecated",
    create_pr: Optional[bool] = False,
    max_shard_size: Union[str, int, None] = None,
    num_shards: Optional[int] = None,
    embed_external_files: bool = True,
) -> huggingface_hub.hf_api.CommitInfo:
    return input_dataset.push_to_hub(
        repo_id,
        config_name,
        set_default,
        split,
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
    input_dataset: Dataset,
    column_names: Union[str, List[str]],
    new_fingerprint: Optional[str] = None,
) -> "Dataset":
    return input_dataset.remove_columns(column_names, new_fingerprint)


def rename_column(
    input_dataset: Dataset,
    original_column_name: str,
    new_column_name: str,
    new_fingerprint: Optional[str] = None,
) -> "Dataset":
    return input_dataset.rename_column(
        original_column_name, new_column_name, new_fingerprint
    )


def rename_columns(
    input_dataset: Dataset,
    column_mapping: Dict[str, str],
    new_fingerprint: Optional[str] = None,
) -> "Dataset":
    return input_dataset.rename_columns(column_mapping, new_fingerprint)


def reset_format(input_dataset: Dataset) -> None:
    return input_dataset.reset_format()


def save_faiss_index(
    input_dataset: Dataset,
    index_name: str,
    file: Union[str, pathlib.PurePath],
    storage_options: Optional[Dict] = None,
) -> None:
    return input_dataset.save_faiss_index(index_name, file, storage_options)


def save_to_disk(
    input_dataset: Dataset,
    dataset_path: Union[str, bytes, os.PathLike],
    fs="deprecated",
    max_shard_size: Union[str, int, None] = None,
    num_shards: Optional[int] = None,
    num_proc: Optional[int] = None,
    storage_options: Optional[dict] = None,
) -> None:
    return input_dataset.save_to_disk(
        dataset_path, fs, max_shard_size, num_shards, num_proc, storage_options
    )


def search(
    input_dataset: Dataset,
    index_name: str,
    query: Union[str, np.array],
    k: int = 10,
    **kwargs,
) -> datasets.search.SearchResults:
    return input_dataset.search(index_name, query, k, kwargs)


def search_batch(
    input_dataset: Dataset,
    index_name: str,
    queries: Union[List[str], np.array],
    k: int = 10,
    **kwargs,
) -> datasets.search.BatchedSearchResults:
    return input_dataset.search_batch(index_name, queries, k, kwargs)


def select(
    input_dataset: Dataset,
    indices: Iterable,
    keep_in_memory: bool = False,
    indices_cache_file_name: Optional[str] = None,
    writer_batch_size: Optional[int] = 1000,
    new_fingerprint: Optional[str] = None,
) -> "Dataset":
    return input_dataset.select(
        indices,
        keep_in_memory,
        indices_cache_file_name,
        writer_batch_size,
        new_fingerprint,
    )


def select_columns(
    input_dataset: Dataset,
    column_names: Union[str, List[str]],
    new_fingerprint: Optional[str] = None,
) -> "Dataset":
    return input_dataset.select_columns(column_names, new_fingerprint)


def set_format(
    input_dataset: Dataset,
    type: Optional[str] = None,
    columns: Optional[List] = None,
    output_all_columns: bool = False,
    **format_kwargs,
) -> None:
    return input_dataset.set_format(type, columns, output_all_columns, format_kwargs)


def set_transform(
    input_dataset: Dataset,
    transform: Optional[Callable],
    columns: Optional[List] = None,
    output_all_columns: bool = False,
) -> None:
    return input_dataset.set_transform(transform, columns, output_all_columns)


def shard(
    input_dataset: Dataset,
    num_shards: int,
    index: int,
    contiguous: bool = False,
    keep_in_memory: bool = False,
    indices_cache_file_name: Optional[str] = None,
    writer_batch_size: Optional[int] = 1000,
) -> "Dataset":
    return input_dataset.shard(
        num_shards,
        index,
        contiguous,
        keep_in_memory,
        indices_cache_file_name,
        writer_batch_size,
    )


def shuffle(
    input_dataset: Dataset,
    seed: Optional[int] = None,
    generator: Optional[numpy.random._generator.Generator] = None,
    keep_in_memory: bool = False,
    load_from_cache_file: Optional[bool] = None,
    indices_cache_file_name: Optional[str] = None,
    writer_batch_size: Optional[int] = 1000,
    new_fingerprint: Optional[str] = None,
) -> "Dataset":
    return input_dataset.shuffle(
        seed,
        generator,
        keep_in_memory,
        load_from_cache_file,
        indices_cache_file_name,
        writer_batch_size,
        new_fingerprint,
    )


def sort(
    input_dataset: Dataset,
    column_names: Union[str, Sequence[str]],
    reverse: Union[bool, Sequence[bool]] = False,
    kind="deprecated",
    null_placement: str = "at_end",
    keep_in_memory: bool = False,
    load_from_cache_file: Optional[bool] = None,
    indices_cache_file_name: Optional[str] = None,
    writer_batch_size: Optional[int] = 1000,
    new_fingerprint: Optional[str] = None,
) -> "Dataset":
    return input_dataset.sort(
        column_names,
        reverse,
        kind,
        null_placement,
        keep_in_memory,
        load_from_cache_file,
        indices_cache_file_name,
        writer_batch_size,
        new_fingerprint,
    )


def to_csv(
    input_dataset: Dataset,
    path_or_buf: Union[str, bytes, os.PathLike, BinaryIO],
    batch_size: Optional[int] = None,
    num_proc: Optional[int] = None,
    **to_csv_kwargs,
) -> int:
    return input_dataset.to_csv(path_or_buf, batch_size, num_proc, to_csv_kwargs)


def to_dict(
    input_dataset: Dataset, batch_size: Optional[int] = None, batched="deprecated"
) -> Union[dict, Iterator[dict]]:
    return input_dataset.to_dict(batch_size, batched)


def to_iterable_dataset(
    input_dataset: Dataset, num_shards: Optional[int] = 1
) -> "IterableDataset":
    return input_dataset.to_iterable_dataset(num_shards)


def to_json(
    input_dataset: Dataset,
    path_or_buf: Union[str, bytes, os.PathLike, BinaryIO],
    batch_size: Optional[int] = None,
    num_proc: Optional[int] = None,
    **to_json_kwargs,
) -> int:
    return input_dataset.to_json(path_or_buf, batch_size, num_proc, to_json_kwargs)


def to_list(input_dataset: Dataset) -> list:
    return input_dataset.to_list()


def to_pandas(
    input_dataset: Dataset, batch_size: Optional[int] = None, batched: bool = False
) -> Union[pandas.core.frame.DataFrame, Iterator[pandas.core.frame.DataFrame]]:
    return input_dataset.to_pandas(batch_size, batched)


def to_parquet(
    input_dataset: Dataset,
    path_or_buf: Union[str, bytes, os.PathLike, BinaryIO],
    batch_size: Optional[int] = None,
    **parquet_writer_kwargs,
) -> int:
    return input_dataset.to_parquet(path_or_buf, batch_size, parquet_writer_kwargs)


def to_sql(
    input_dataset: Dataset,
    name: str,
    con: Union[
        str,
        ForwardRef("sqlalchemy.engine.Connection"),
        ForwardRef("sqlalchemy.engine.Engine"),
        ForwardRef("sqlite3.Connection"),
    ],
    batch_size: Optional[int] = None,
    **sql_writer_kwargs,
) -> int:
    return input_dataset.to_sql(name, con, batch_size, sql_writer_kwargs)


def to_tf_dataset(
    input_dataset: Dataset,
    batch_size: Optional[int] = None,
    columns: Union[str, List[str], None] = None,
    shuffle: bool = False,
    collate_fn: Optional[Callable] = None,
    drop_remainder: bool = False,
    collate_fn_args: Optional[Dict[str, Any]] = None,
    label_cols: Union[str, List[str], None] = None,
    prefetch: bool = True,
    num_workers: int = 0,
    num_test_batches: int = 20,
) -> None:
    return input_dataset.to_tf_dataset(
        batch_size,
        columns,
        shuffle,
        collate_fn,
        drop_remainder,
        collate_fn_args,
        label_cols,
        prefetch,
        num_workers,
        num_test_batches,
    )


def train_test_split(
    input_dataset: Dataset,
    test_size: Union[float, int, None] = None,
    train_size: Union[float, int, None] = None,
    shuffle: bool = True,
    stratify_by_column: Optional[str] = None,
    seed: Optional[int] = None,
    generator: Optional[numpy.random._generator.Generator] = None,
    keep_in_memory: bool = False,
    load_from_cache_file: Optional[bool] = None,
    train_indices_cache_file_name: Optional[str] = None,
    test_indices_cache_file_name: Optional[str] = None,
    writer_batch_size: Optional[int] = 1000,
    train_new_fingerprint: Optional[str] = None,
    test_new_fingerprint: Optional[str] = None,
) -> "DatasetDict":
    return input_dataset.train_test_split(
        test_size,
        train_size,
        shuffle,
        stratify_by_column,
        seed,
        generator,
        keep_in_memory,
        load_from_cache_file,
        train_indices_cache_file_name,
        test_indices_cache_file_name,
        writer_batch_size,
        train_new_fingerprint,
        test_new_fingerprint,
    )


def unique(input_dataset: Dataset, column: str) -> List:
    return input_dataset.unique(column)


def with_format(
    input_dataset: Dataset,
    type: Optional[str] = None,
    columns: Optional[List] = None,
    output_all_columns: bool = False,
    **format_kwargs,
) -> None:
    return input_dataset.with_format(type, columns, output_all_columns, format_kwargs)


def with_transform(
    input_dataset: Dataset,
    transform: Optional[Callable],
    columns: Optional[List] = None,
    output_all_columns: bool = False,
) -> None:
    return input_dataset.with_transform(transform, columns, output_all_columns)
