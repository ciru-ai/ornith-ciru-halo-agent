"""Target-only CPU semantic phases for the isolated IU4 correctness candidate.

This does not infer phase from tensor height. A prompt can have one row and
speculative verification can have many rows. FULL uniform capture is decode;
ordinary dummy profiling is prefill. Real padding retains the parent prefill
policy. No GPU metadata reads, cache edits, or scheduler changes are made.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps


@dataclass(frozen=True)
class Plan:
    requests: tuple
    segments: tuple
    rows: int


def plan_from_requests(request_ids, counts, computed, prompts):
    if not (len(request_ids) == len(counts) == len(computed) == len(prompts)):
        raise ValueError("IU4 phase metadata lengths differ")
    if not request_ids or len(set(request_ids)) != len(request_ids):
        raise ValueError("IU4 phases require unique nonempty request IDs")
    segments, requests, offset = [], [], 0

    def append(first, end, prefill):
        if first == end:
            return
        if segments and segments[-1][2] == prefill:
            segments[-1] = (segments[-1][0], end, prefill)
        else:
            segments.append((first, end, prefill))

    for req, count, done, prompt in zip(request_ids, counts, computed, prompts):
        count, done, prompt = int(count), int(done), int(prompt)
        if count <= 0 or done < 0 or prompt <= 0:
            raise ValueError("Invalid IU4 semantic phase counts")
        boundary = offset + min(count, max(prompt - done, 0))
        append(offset, boundary, True)
        append(boundary, offset + count, False)
        requests.append((req, count, done, prompt))
        offset += count
    return Plan(tuple(requests), tuple(segments), offset)


_STATE = ("synthetic", None)
_INSTALLED = False


@contextmanager
def scope(kind, value=None):
    global _STATE
    if kind not in ("synthetic", "pending", "real", "capture"):
        raise ValueError("Unknown IU4 semantic phase scope")
    if kind == "real" and not isinstance(value, Plan):
        raise ValueError("A real IU4 phase scope needs an admitted CPU plan")
    if kind == "capture" and (not isinstance(value, int) or value <= 0):
        raise ValueError("An IU4 decode capture needs a positive row count")
    previous, _STATE = _STATE, (kind, value)
    try:
        yield
    finally:
        _STATE = previous


def segments_for(rows):
    kind, value = _STATE
    if rows <= 0:
        raise ValueError("Empty IU4 projection")
    if kind == "pending":
        raise RuntimeError("A real IU4 projection has no prepared CPU phase plan")
    if kind == "capture":
        if rows != value:
            raise ValueError("IU4 projection rows differ from its uniform capture descriptor")
        return ((0, rows, False),)
    if kind == "synthetic":
        return ((0, rows, True),)
    if value.rows > rows:
        raise ValueError("IU4 projection truncates admitted CPU phase rows")
    result = value.segments
    if value.rows < rows:
        result += ((value.rows, rows, True),)
    return result


def install():
    """Install after the released worker's own phase/dynamic graph hooks."""
    global _INSTALLED, _STATE
    if _INSTALLED:
        return
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    cls = GPUModelRunner
    original_prepare = cls._prepare_inputs
    original_execute = cls.execute_model
    original_capture = cls._warmup_and_capture
    original_dummy = cls._dummy_run
    original_determine = cls._determine_batch_execution_and_padding

    @wraps(original_prepare)
    def prepare(runner, scheduler_output, num_scheduled_tokens):
        global _STATE
        if runner.use_async_scheduling:
            raise ValueError("IU4 correctness phases require synchronous scheduling")
        batch, count = runner.input_batch, runner.input_batch.num_reqs
        plan = plan_from_requests(tuple(batch.req_ids[:count]), tuple(num_scheduled_tokens),
            tuple(batch.num_computed_tokens_cpu[:count]), tuple(batch.num_prompt_tokens[:count]))
        if sum(scheduler_output.num_scheduled_tokens.values()) != plan.rows:
            raise ValueError("IU4 phase plan differs from scheduler token total")
        result = original_prepare(runner, scheduler_output, num_scheduled_tokens)
        _STATE = ("real", plan)
        return result

    @wraps(original_execute)
    def execute(runner, *args, **kwargs):
        with scope("pending"):
            return original_execute(runner, *args, **kwargs)

    @wraps(original_capture)
    def capture(runner, desc, cudagraph_runtime_mode, *args, **kwargs):
        uniform_full = getattr(cudagraph_runtime_mode, "name", None) == "FULL" and desc.uniform
        with scope("capture", int(desc.num_tokens)) if uniform_full else scope("synthetic"):
            return original_capture(runner, desc, cudagraph_runtime_mode, *args, **kwargs)

    @wraps(original_dummy)
    def dummy(runner, *args, **kwargs):
        # _warmup_and_capture includes NONE warmups before the FULL capture.
        # They must warm exactly the same decode projection policy.
        if _STATE[0] == "capture":
            return original_dummy(runner, *args, **kwargs)
        with scope("synthetic"):
            return original_dummy(runner, *args, **kwargs)

    @wraps(original_determine)
    def determine(runner, num_tokens, num_reqs, num_scheduled_tokens_np,
                  max_num_scheduled_tokens, use_cascade_attn, *args, **kwargs):
        result = original_determine(runner, num_tokens, num_reqs, num_scheduled_tokens_np,
            max_num_scheduled_tokens, use_cascade_attn, *args, **kwargs)
        if kwargs.get("force_uniform_decode") is None and num_tokens > 0:
            kind, plan = _STATE
            if kind != "real" or plan.rows != num_tokens:
                raise RuntimeError("Real IU4 graph dispatch lacks a current semantic phase plan")
            if getattr(result[0], "name", None) == "FULL":
                if any(prefill for _, _, prefill in plan.segments):
                    raise RuntimeError("IU4 FULL decode graph admitted prompt rows")
                if result[1].num_tokens < num_tokens:
                    raise RuntimeError("IU4 FULL graph truncates scheduled token rows")
        return result

    cls._prepare_inputs = prepare
    cls.execute_model = execute
    cls._warmup_and_capture = capture
    cls._dummy_run = dummy
    cls._determine_batch_execution_and_padding = determine
    _INSTALLED = True
