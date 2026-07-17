"""Convenience wrappers for clearML."""

from collections.abc import Sequence
from pathlib import Path

from clearml import Dataset, Task, TaskTypes

PROJECT = "html_highlight"


def task_init(task_name: str, task_type: TaskTypes = TaskTypes.training):
    task = Task.init(PROJECT, task_name, task_type)
    assert isinstance(task, Task)
    return task


def create_dataset(
    files: list[Path],
    name: str,
    project: str = PROJECT,
    parents: Sequence[str | Dataset] | None = None,
):
    dataset = Dataset.create(dataset_project=project, dataset_name=name, parent_datasets=parents)

    for f in files:
        dataset.add_files(path=f)

    # Upload dataset to ClearML server
    dataset.upload()

    # commit dataset changes
    dataset.finalize()

    return dataset
