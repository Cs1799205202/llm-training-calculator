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
WHERE iteration > 0 AND
    micro_batch_index = 0
    AND (
        slice.name = "send-warmup"
        OR slice.name = "send-extra"
    )
ORDER BY iteration, pid;

SELECT * FROM slice WHERE EXTRACT_ARG(arg_set_id, "args.iteration") = 0 AND name = "optimizer";