from pathlib import Path
from typing import List
from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig

def detect_in_pipeline_parallel_group(
    trace: str | TraceProcessor,
    comm_group: List[int],
    start_iteration: int,
):
    if isinstance(trace, str):
        bin_path = Path().cwd() / "trace_processor"
        print(f"bin_path: {bin_path}")
        tp = TraceProcessor(trace=trace, config=TraceProcessorConfig(bin_path=bin_path.as_posix(), verbose=True))
    sql: str = f"""
SELECT slice.id AS id,
    slice.name AS name,
    EXTRACT_ARG(slice.arg_set_id, "args.iteration") AS iteration,
    process.pid AS pid,
    thread.tid AS tid,
    EXTRACT_ARG(slice.arg_set_id, "args.micro_batch_index") AS micro_batch_index,
    slice.dur AS duration,
    EXTRACT_ARG(slice.arg_set_id, "args.data") AS data,
    EXTRACT_ARG(slice.arg_set_id, "args.bandwidth") AS bandwidth
FROM slice
    JOIN thread_track ON thread_track.id = slice.track_id
    JOIN thread USING (utid)
    JOIN process USING (upid)
WHERE iteration > {start_iteration} AND
    micro_batch_index = 0
    AND (
        slice.name = "send-warmup"
        OR slice.name = "send-extra"
    )
ORDER BY iteration, pid
"""
    qr_it = tp.query(sql)
    for row in qr_it:
        print(row)


if __name__ == "__main__":
    detect_in_pipeline_parallel_group(
        trace="benchmark.json",
        comm_group=[0, 1, 2, 3, 4, 5, 6, 7],
        start_iteration=1,
    )