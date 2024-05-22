import csv
from dataclasses import dataclass
import json
import os
from typing import Any, List, Tuple


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


def collect_benchmark_files(dir: os.PathLike) -> List[Tuple[Rank, List[str]]]:
    """Collect benchmark-rank-*.csv files from the given directory."""
    files = []
    for file in os.listdir(dir):
        if file.startswith('benchmark-') and file.endswith('.csv'):
            desc = file[len('benchmark-'):-len('.csv')]
            # disc is "data-*-tensor-*-pipeline-*"
            fields = desc.split('-')
            chunks = dict((fields[i], int(fields[i + 1])) for i in range(0, len(fields), 2))
            rank = Rank(**chunks)
            with open(os.path.join(dir, file), 'r') as f:
                files.append((rank, f.readlines()))
    return files


@dataclass
class Event:
    timestamp: float
    rank: Rank
    name: str
    predicate: str


@dataclass
class Iteration:
    events: List[Event]
    duration: float


def read_benchmark_file(rank: Rank, content: List[str]) -> List[Iteration]:
    """Returns events in each iteration."""
    data = []
    reader = csv.reader(content)
    for row in reader:
        if row[0] == 'iteration start':
            current_iteration = []
            timeline = 0.0
        elif row[0] == 'iteration end':
            data.append(Iteration(events=current_iteration, duration=timeline))
            current_iteration = None
            timeline = None
        else:
            if timeline is None:
                # In evaluation, so ignore.
                continue
            what, delta = row
            # what is "event-name predicate"
            name, predicate = what.split(' ')
            delta = float(delta)
            timeline += delta
            event = Event(timestamp=timeline, rank=rank, name=name, predicate=predicate)
            current_iteration.append(event)
    return data


def aggregate_benchmark_data(contents: List[List[Iteration]]) -> List[Iteration]:
    """Sort and aggregate benchmark data."""
    num_iterations = len(contents[0])
    assert all(len(content) == num_iterations for content in contents), 'Mismatched number of iterations'

    iterations = []

    for i in range(num_iterations):
        events = [event for content in contents for event in content[i].events]
        events.sort(key=lambda event: event.timestamp)
        duration = max(content[i].duration for content in contents)
        iterations.append(Iteration(events=events, duration=duration))

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
    'warmup-forward': COLOR_FORWARD,
    'backward': COLOR_BACKWARD,
    'cooldown-backward': COLOR_BACKWARD,
    'recv-extra': COLOR_RECV,
    'warmup-recv': COLOR_RECV,
    'cooldown-recv': COLOR_RECV,
    'send-extra': COLOR_SEND,
    'warmup-send': COLOR_SEND,
    'cooldown-send': COLOR_SEND,
    'exchange-next': COLOR_EXCHANGE_NEXT,
    'exchange-prev': COLOR_EXCHANGE_PREV,
    'allreduce': COLOR_ALLREDUCE,
    'optimizer': COLOR_OPTIMIZER,
}

def benchmark_to_chrome_trace(iterations: List[Iteration]) -> Any:
    """Convert benchmark data to Chrome trace format."""
    pipeline_paralellism = max(event.rank.pipeline for iteration in iterations for event in iteration.events) + 1
    trace = []
    timeline = 0.0
    for i, iteration in enumerate(iterations):
        for event in iteration.events:
            if event.predicate == 'start':
                phase = 'B'
            elif event.predicate == 'end':
                phase = 'E'
            else:
                phase = 'i'
            trace.append({
                'name': event.name,
                'cname': COLOR_MAP.get(event.name, COLOR_UNKNOWN),
                'cat': 'benchmark',
                'ph': phase,
                'ts': int((event.timestamp + timeline) * 1e6),
                'pid': event.rank.to_pid(pipeline_paralellism),
                'tid': event.rank.to_tid(),
                # iteration number
                'args': {'iteration': i}
            })
        timeline += iteration.duration
    return trace


if __name__ == "__main__":
    benchmark_dir = '.'
    files = collect_benchmark_files(benchmark_dir)
    contents = [read_benchmark_file(rank, content) for rank, content in files]
    aggregated = aggregate_benchmark_data(contents)
    with open('benchmark.csv', 'w') as f:
        writer = csv.writer(f)
        for iteration in aggregated:
            for event in iteration.events:
                writer.writerow([event.timestamp, str(event.rank), event.name, event.predicate])
            writer.writerow([])
    with open('benchmark.json', 'w') as f:
        json.dump(benchmark_to_chrome_trace(aggregated), f)
