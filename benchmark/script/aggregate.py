from argparse import Namespace
from dataclasses import dataclass
import json
from pathlib import Path
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
from sklearn.cluster import DBSCAN

logging.basicConfig(level=logging.DEBUG)

@dataclass
class Rank:
    data: int
    pipeline: int
    tensor: int

    def __str__(self) -> str:
        return f'{self.data}-{self.pipeline}-{self.tensor}'
    
    def to_pid(self, pipeline_paralellism: int) -> int:
        return self.data * pipeline_paralellism + self.pipeline
    
    def to_tid(self) -> int:
        return self.tensor


def collect_benchmark_files(dir: os.PathLike) -> List[Tuple[Rank, str]]:
    """Collect benchmark.json files from the given directory."""
    files = []
    with os.scandir(dir) as it:
        for entry in it:
            file = entry.name
            if file.startswith("benchmark-") and file.endswith(".json"):
                desc = file[len("benchmark-") : -len(".json")]
                # disc is "data-*-pipeline-*-tensor-*"
                fields = desc.split("-")
                chunks = dict(
                    (fields[i], int(fields[i + 1])) for i in range(0, len(fields), 2)
                )
                rank = Rank(**chunks)
                with open(os.path.join(dir, file), "r") as f:
                    files.append((rank, f.read()))
    return files


@dataclass
class Event:
    rel_ts: int
    rank: Rank
    name: str
    ph: str
    attrs: Any
    cat: Optional[str] = None


@dataclass
class Iteration:
    pad_before: int
    events: List[Event]
    duration: int


def read_benchmark_file(rank: Rank, content: str) -> List[Iteration]:
    """Returns events in each iteration."""
    data = []
    rows: List[Dict[str, Any]] = json.loads(content)
    for row in rows:
        if row['name'] == 'iteration' and row['ph'] == 'B':
            pad_before = row['pad_before']
            current_iteration = []
        elif row['name'] == 'iteration' and row['ph'] == 'E':
            duration = row['duration_wall']
            data.append(
                Iteration(
                    pad_before=pad_before if pad_before is not None else 0,
                    events=current_iteration if current_iteration is not None else [],
                    duration=duration,
                )
            )
            pad_before = None
            current_iteration = None
        else:
            if current_iteration is None:
                # In evaluation, so ignore.
                continue
            name = row['name']
            rel_ts = row['rel_ts']
            ph = row['ph']
            del row['name']
            del row['rel_ts']
            del row['ph']
            cat = None
            if row.get('cat') is not None:
                cat = row['cat']
                del row['cat']
            event = Event(rel_ts=rel_ts, rank=rank, name=name, ph=ph, attrs=row, cat=cat)
            current_iteration.append(event)
    return data


TP_ALIGNMENT_EVENTS: List[str] = ['_reduce']
PP_ALIGNMENT_EVENTS: List[Tuple[str, str]] = [('recv_backward', 'recv_forward')]

DATA_PARALLELISM: int
PIPELINE_PARALLELISM: int
TENSOR_PARALLELISM: int

def aggregate_benchmark_data(contents: List[List[Iteration]]) -> List[Iteration]:
    """Sort and aggregate benchmark data."""
    num_iterations = len(contents[0])
    assert all(len(content) == num_iterations for content in contents), 'Mismatched number of iterations'

    global DATA_PARALLELISM, PIPELINE_PARALLELISM, TENSOR_PARALLELISM
    DATA_PARALLELISM = max(event.rank.data for content in contents for event in content[0].events) + 1
    PIPELINE_PARALLELISM = max(event.rank.pipeline for content in contents for event in content[0].events) + 1
    TENSOR_PARALLELISM = max(event.rank.tensor for content in contents for event in content[0].events) + 1
    logging.debug(f'data_parallelism={DATA_PARALLELISM}, pipeline_paralellism={PIPELINE_PARALLELISM}, tensor_parallelism={TENSOR_PARALLELISM}')

    # temp: List[List[List[List[List[Event]]]]] = [
    #     [[[[]] * num_iterations] * tensor_parallelism] * pipeline_paralellism
    # ] * data_parallelism
    # * only does shallow copy
    temp: List[List[List[List[List[Event]]]]] = [
        [
            [
                [[] for _ in range(num_iterations)]
                for _ in range(TENSOR_PARALLELISM)
            ]
            for _ in range(PIPELINE_PARALLELISM)
        ]
        for _ in range(DATA_PARALLELISM)
    ]
    for content in contents:
        for i in range(num_iterations):
            # logging.debug(f'content[{i}].events={len(content[i].events)}')
            for event in content[i].events:
                temp[event.rank.data][event.rank.pipeline][event.rank.tensor][i].append(event)
                # logging.debug(f'{event.rank.data} {event.rank.pipeline} {event.rank.tensor} {i} {len(temp[0][0][0][i])}')
            # logging.debug(f'{num_iterations} {i} {len(temp[0][0][1][i])}')

    # align in tensor parallelism group
    for i in range(DATA_PARALLELISM):
        for j in range(PIPELINE_PARALLELISM):
            for iteration in range(num_iterations):
                # use the first rank of the tensor parallelism group as the reference
                ref_events: List[Event] = temp[i][j][0][iteration]
                for k in range(1, TENSOR_PARALLELISM):
                    events: List[Event] = temp[i][j][k][iteration]
                    assert len(ref_events) == len(events), 'Mismatched number of events'
                    offset: int = 0
                    for l in range(len(ref_events)):
                        assert ref_events[l].name == events[l].name, 'Mismatched event name'
                        if ref_events[l].name in TP_ALIGNMENT_EVENTS and ref_events[l].ph == 'E':
                            offset = ref_events[l].rel_ts - events[l].rel_ts
                            events[l - 1].rel_ts += offset
                        events[l].rel_ts += offset
    
    # align in pipeline parallelism group
    # In a pipeline parallelism group, the current rank's 'recv_backward' should be aligned with the next rank's 'recv_forward'
    # e.g. dp=2, pp=2, tp=2:
    # rank[0-0-0] 'recv_backward' should be aligned with rank[0-1-0] 'recv_forward'
    # rank[0-0-1] 'recv_backward' should be aligned with rank[0-1-1] 'recv_forward'
    # Note that the index of events in 'recv_backward' and 'recv_forward' of consecutive ranks may not be the same
    # TODO 

    # for content in contents:
    #     for i in range(num_iterations):
    #         # logging.debug(f'content[{i}].events={len(content[i].events)}, temp={len(temp[0][0][0][i])}')
    #         for idx, event in enumerate(content[i].events):
    #             event.rel_ts = temp[event.rank.data][event.rank.pipeline][event.rank.tensor][i][idx].rel_ts

    iterations: List[Iteration] = []

    for i in range(num_iterations):
        pad_before: int = max(content[i].pad_before for content in contents)
        events = [event for content in contents for event in content[i].events]
        events.sort(key=lambda event: event.rel_ts)
        duration: int = max(content[i].duration for content in contents)
        iterations.append(Iteration(pad_before=pad_before, events=events, duration=duration))

    return iterations

COLOR_UNKNOWN = 'thread_state_unknown'
COLOR_FORWARD = 'thread_state_running'
COLOR_BACKWARD = 'thread_state_iowait'
COLOR_RECV = 'rail_response'
COLOR_SEND = 'rail_animation'
COLOR_EXCHANGE_NEXT = 'thread_state_runnable'
COLOR_EXCHANGE_PREV = 'thread_state_sleeping'
COLOR_ALLREDUCE = 'light_memory_dump'
COLOR_OPTIMIZER = 'detailed_memory_dump'
COLOR_MAP = {
    'forward': COLOR_FORWARD,
    'forward-warmup': COLOR_FORWARD,
    'backward': COLOR_BACKWARD,
    'backward-cooldown': COLOR_BACKWARD,
    'recv-extra': COLOR_RECV,
    'recv-warmup': COLOR_RECV,
    'recv-forward': COLOR_RECV,
    'recv-backward': COLOR_RECV,
    'recv-cooldown': COLOR_RECV,
    'send-extra': COLOR_SEND,
    'send-warmup': COLOR_SEND,
    'send-forward': COLOR_SEND,
    'send-backward': COLOR_SEND,
    'send-cooldown': COLOR_SEND,
    'exchange-next': COLOR_EXCHANGE_NEXT,
    'exchange-prev': COLOR_EXCHANGE_PREV,
    'allreduce': COLOR_ALLREDUCE,
    'optimizer': COLOR_OPTIMIZER,
}

def benchmark_to_chrome_trace(iterations: List[Iteration]) -> Any:
    """Convert benchmark data to Chrome trace format."""
    pipeline_paralellism = max(event.rank.pipeline for iteration in iterations for event in iteration.events) + 1
    traces = []
    timeline = 0
    for i, iteration in enumerate(iterations):
        timeline += iteration.pad_before
        for event in iteration.events:
            trace = {
                'name': event.name,
                'cname': COLOR_MAP.get(event.name, COLOR_UNKNOWN),
                'ph': event.ph,
                'ts': int((event.rel_ts + timeline) / 1e3),
                'pid': event.rank.to_pid(pipeline_paralellism),
                'tid': event.rank.to_tid(),
                # iteration number
                'args': {'iteration': i, **event.attrs}
            }
            if event.cat is not None:
                trace['cat'] = event.cat
            traces.append(trace)
        timeline += iteration.duration
    return traces

def try_detect(contents: List[List[Iteration]], method: str = "naive") -> Optional[Rank]:
    # durations [data][pipeline][tensor][]
    durations: List[List[List[List[Dict[str, Any]]]]] = [
        [
            [
                []
                for _ in range(TENSOR_PARALLELISM)
            ]
            for _ in range(PIPELINE_PARALLELISM)
        ]
        for _ in range(DATA_PARALLELISM)
    ]

    for content in contents:
        rank: Rank = content[0].events[0].rank
        logging.debug(f'rank={rank}')
        assert all(rank == event.rank for iteration in content for event in iteration.events), 'Mismatched rank'
        start_times = {}
        for i in range(len(content)):
            iteration: Iteration = content[i]
            for event in iteration.events:
                name = event.name
                if event.ph == 'B':
                    start_times[name] = event.rel_ts
                elif event.ph == 'E':
                    if name in start_times:
                        duration = event.rel_ts - start_times[name]
                        durations[rank.data][rank.pipeline][rank.tensor].append({
                            'iteration': i,
                            'name': name,
                            'duration': duration,
                            'rank': rank
                        })
    suspects: List[Rank] = []
    for data in durations:
        for pipeline in data:
            event_cnt: int = len(pipeline[0])
            for tensor in pipeline:
                logging.debug(f'len(tensor)={len(tensor)}, event_cnt={event_cnt}')
            assert all(len(tensor) == event_cnt for tensor in pipeline), 'Mismatched number of events'
            for i in range(event_cnt):
                event_name = pipeline[0][i]['name']
                assert all(tensor[i]['name'] == event_name for tensor in pipeline), 'Mismatched event name'
                times: List[Tuple[int, Rank]] = [(tensor[i]['duration'], tensor[i]['rank']) for tensor in pipeline]
                times.sort(key=lambda x: x[0])
                assume_outlier: Tuple[int, Rank] = times[0]
                times.pop(0)
                avg = np.mean([x[0] for x in times])
                std = np.std([x[0] for x in times])
                if assume_outlier[0] < avg - 3 * std:
                    suspects.append(assume_outlier[1])
                    # logging.info(f'Assume outlier: {assume_outlier[1]} {event_name} {assume_outlier[0]} {avg} {avg - 3 * std} {[x[0] for x in times]}')
    import collections
    logging.info(f'{collections.Counter(map(str, suspects))}')
    for i_data, data in enumerate(durations):
        for i_pipeline, pipeline in enumerate(data):
            for i_tensor, tensor in enumerate(pipeline):
                logging.debug(f'{i_data}-{i_pipeline}-{i_tensor} {len(tensor)}')
    abnormal_rank: str = collections.Counter(map(str, suspects)).most_common(1)[0][0]
    # TODO: convert rank to index of device
    return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--bench-dir",
        type=Path,
        default=Path.cwd() / "Megatron",
        help="directory containing benchmark files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("benchmark.json"),
        help="output chrome trace file",
    )
    parser.add_argument(
        "-d",
        "--detect",
        action="store_true",
        default=False,
        help="try to detect the abnormal GPU, now assume only one GPU is abnormal",
    )
    args = parser.parse_args()
    benchmark_dir: Path = args.bench_dir
    files = collect_benchmark_files(benchmark_dir)
    contents = [read_benchmark_file(rank, content) for rank, content in files]
    aggregated: List[Iteration] = aggregate_benchmark_data(contents)
    with open(args.output, 'w') as f:
        json.dump(benchmark_to_chrome_trace(aggregated), f, indent=2)

    if args.detect:
        try_detect(contents)
        pass
