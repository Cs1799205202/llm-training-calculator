from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Optional, Tuple


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
    for file in os.listdir(dir):
        if file.startswith('benchmark-') and file.endswith('.json'):
            desc = file[len('benchmark-'):-len('.json')]
            # disc is "data-*-pipeline-*-tensor-*"
            fields = desc.split('-')
            chunks = dict((fields[i], int(fields[i + 1])) for i in range(0, len(fields), 2))
            rank = Rank(**chunks)
            with open(os.path.join(dir, file), 'r') as f:
                files.append((rank, f.read()))
    return files


@dataclass
class Event:
    timestamp: int
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
            pad_before = row['delta']
            current_iteration = []
            timeline = 0
        elif row['name'] == 'iteration' and row['ph'] == 'E':
            data.append(Iteration(pad_before=pad_before, events=current_iteration, duration=timeline))
            pad_before = None
            current_iteration = None
            timeline = None
        else:
            if timeline is None:
                # In evaluation, so ignore.
                continue
            name = row['name']
            delta = row['delta']
            ph = row['ph']
            del row['name']
            del row['delta']
            del row['ph']
            cat = None
            if row.get('cat') is not None:
                cat = row['cat']
                del row['cat']
            timeline += delta
            event = Event(timestamp=timeline, rank=rank, name=name, ph=ph, attrs=row, cat=cat)
            current_iteration.append(event)
    return data


def aggregate_benchmark_data(contents: List[List[Iteration]]) -> List[Iteration]:
    """Sort and aggregate benchmark data."""
    num_iterations = len(contents[0])
    assert all(len(content) == num_iterations for content in contents), 'Mismatched number of iterations'

    iterations = []

    for i in range(num_iterations):
        pad_before = max(content[i].pad_before for content in contents)
        events = [event for content in contents for event in content[i].events]
        events.sort(key=lambda event: event.timestamp)
        duration = max(content[i].duration for content in contents)
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
    traces = []
    timeline = 0
    for i, iteration in enumerate(iterations):
        timeline += iteration.pad_before
        for event in iteration.events:
            trace = {
                'name': event.name,
                'cname': COLOR_MAP.get(event.name, COLOR_UNKNOWN),
                'ph': event.ph,
                'ts': int((event.timestamp + timeline) / 1e3),
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


if __name__ == "__main__":
    benchmark_dir = '.'
    files = collect_benchmark_files(benchmark_dir)
    if len(files) == 0:
        files = collect_benchmark_files(os.path.join(benchmark_dir, 'Megatron'))
    contents = [read_benchmark_file(rank, content) for rank, content in files]
    aggregated = aggregate_benchmark_data(contents)
    with open('benchmark.json', 'w') as f:
        json.dump(benchmark_to_chrome_trace(aggregated), f, indent=2)
