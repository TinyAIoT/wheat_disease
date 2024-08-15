import numpy as np
from typing import List, Union, Callable, Literal, Dict
from collections import Counter
import os
import json
from ei_shared.types import ClassificationMode


def allocate_to_bins(array: List, num_bins: int):
    """Given an array, and a number of bins, return a grouping across bins.

    The core expected use of this method being determining a grouping
    for continuous values in regression metrics. e.g.

    groups = allocate_to_bins(y_pred, num_bins=3)
    calculate_regression_metrics(y_true, y_pred, groups=groups)

    Args:
        array: a list of numerical values
        num_bins: the number of bins to allocate across
    Returns:
        a grouping, the same length as input array, that can be used as the
        `groups` are for calculate_regression_metrics
    """

    array = np.array(array)

    # calculate the bin edges for a N bin histogram
    _bin_allocation, bin_edges = np.histogram(array, bins=num_bins)

    # allocate array values to groups based on bin_edges; i.e. the lowest
    # elements are in group 1, next elements are in group 2 etc.
    # since digitize uses >= this though results in the last bin having only
    # the max element. to avoid this we can ignore the last bin_edge; then
    # the max elements is dropped into the previous bin
    groups = np.digitize(array, bin_edges[:-1])

    # convert from numerical index into a human readable range
    # e.g. instead of, say, group=5 we have "(0.56, 0.67)"
    human_readable_groups = []
    for g in groups:
        range_min = bin_edges[g - 1]
        range_max = bin_edges[g]
        human_readable_groups.append(f"({range_min}, {range_max})")

    return human_readable_groups


def calculate_grouped_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    metrics_fn: Callable,
    groups: List,
    max_groups: int = None,
):
    """Given a metrics_fn and a grouping run the metrics_fn for each group.

    Args:
        y_true: complete set of y_true values, as either a list or ndarray
        y_pred: complete set of y_pred values, as either a list or ndarray
        metrics_fn: a callable that returns a dict of metrics for a (sub)set of
            y_true, y_pred values.
        groups: a list of items, the same length as y_true & y_pred that is
            used as a grouping key for metrics_fn calls
        max_groups: the maximum number of grouping that can be returned.
            included so code is robust to a (potentially) large number of
            distinct groups

    Returns:
        A dictionary where metrics_fn has been called for the entire
        y_true, y_pred set as well as subsets of these based on the groups

    E.g. for arguments
        y_true = [3, 1, 4, 1, 5, 9]
        y_pred = [2, 6, 5, 3, 5, 8]
        metrics_fn = lambda(yt, yp): { 'max_t': max(yt), 'max_p': max(yp) }
        groups = ['a', 'a', 'a', 'b', 'c', 'c']

    the return would be
        {'all': { 'max_t': 9, 'max_p': 8 },
         'per_group': {
            'a': { 'max_t': 4, 'max_p': 6 },
            'b': { 'max_t': 1, 'max_p': 3 },
            'c': { 'max_t': 9, 'max_p': 8 },
         }
        }

    additionally if call was made with max_groups=2 then the entry for 'b'
    would not be included since the top 2 elements by frequency are 'a' & 'c'

    """

    # check sizes
    if len(y_true) != len(y_pred) or len(y_true) != len(groups):
        raise Exception(
            "Expected lengths of y_true, y_pred and groups to be the"
            f" same but were {len(y_true)}, {len(y_pred)} and"
            f" {len(groups)} respectively"
        )

    # init returned metrics with an 'all' value
    metrics = {"all": metrics_fn(y_true, y_pred), "per_group": {}}

    if max_groups is None:
        # no max_groups => use all groups
        filtered_groups = set(groups)
    else:
        # determine top groups by element frequency
        group_top_freqs = Counter(groups).most_common(max_groups)
        filtered_groups = [g for g, _freq in group_top_freqs]
        if set(groups) != set(filtered_groups):
            print(
                f"WARNING: filtering from {len(set(groups))} distinct groups"
                f" down to {len(set(filtered_groups))}"
            )

    # when y_true or y_pred are nd arrays we can efficiently slice out
    # a set of indexes with advanced indexing, otherwise we need to index
    # them out explicitly
    def extract_subset(a, idxs):
        if type(a) == np.ndarray:
            return a[idxs]
        elif type(a) == list:
            return [a[i] for i in idxs]
        else:
            raise TypeError(f"Expected ndarray or list, not {type(a)}")

    groups = np.array(groups)
    for group in filtered_groups:
        idxs = np.where(groups == group)[0]
        y_true_subset = extract_subset(y_true, idxs)
        y_pred_subset = extract_subset(y_pred, idxs)
        metrics["per_group"][group] = metrics_fn(y_true_subset, y_pred_subset)

    return metrics


class MetricsJson(object):
    """Helper responsible for shared profiling and testing metrics json"""

    CURRENT_VERSION = 5

    def __init__(self,
                 mode: ClassificationMode,
                 filename: str = "/home/metrics.json",
                 reset: bool=False):

        self.filename = filename
        self.mode = mode

        if not reset and os.path.exists(self.filename):
            try:
                with open(self.filename, "r") as f:
                    self.data = json.load(f)

                if not "version" in self.data:
                    raise Exception('Missing "version"')
                if self.data["version"] == 1:
                    raise Exception("Cannot handle version 1")

            except Exception as e:
                self.data = {"version": MetricsJson.CURRENT_VERSION}
        else:
            self.data = {"version": MetricsJson.CURRENT_VERSION}

        # INVALIDATION LOGIC!
        # if you update stuff here you also need to update 'getAdditionalMetrics' in
        # studio/server/training/learn-helper.ts

        # v4 => #10342 => incorrect conversion of sample dicts for obj detection
        # invalidate all object detection metrics
        if self.data['version'] < 5 and mode == 'object-detection':
            self.data = {
                'version': MetricsJson.CURRENT_VERSION
            }

        # if version is one of the following we cannot trust the test data,
        # v2 => #10261 => incorrect calc for regression
        # v3 => #10262 => i8 overwrites f32 metrics data
        # so invalidate all those test data
        if self.data['version'] in [2, 3]:
            if ('test' in self.data):
                del self.data['test']
            self.data['version'] = MetricsJson.CURRENT_VERSION

    def set(
        self,
        split: Literal["validation", "test"],
        model_type: Literal["float32", "int8"],
        metrics: Dict,
    ):
        if not split in self.data:
            self.data[split] = {}
        self.data[split][model_type] = metrics

    def flush(self):
        with open(self.filename, "w") as f:
            json.dump(self.data, fp=f, indent=4)
