"""R03: preserve requested continuous usage while a strict parser buffers data."""
import hashlib

BASELINE_SHA256 = 'd121657ce64282509d37ce483b257921eb650c776134f72d882067c10abe4eb2'
OLD = '''                        if output.finish_reason is None and (
                            not request.return_token_ids or hide_stream_metadata
                        ):
                            continue
                        delta_message = DeltaMessage()
'''
NEW = '''                        # A parser may buffer a whole JSON array/object.
                        # Real token progress is independent of argument deltas.
                        # Preserve requested usage without exposing partial calls
                        # or hidden reasoning metadata. Empty prefill stays quiet.
                        if (
                            output.finish_reason is None
                            and not (include_continuous_usage and output.token_ids)
                            and (not request.return_token_ids or hide_stream_metadata)
                        ):
                            continue
                        delta_message = DeltaMessage()
'''

def patch_source(source: str) -> str:
    actual = hashlib.sha256(source.encode()).hexdigest()
    if actual != BASELINE_SHA256 or source.count(OLD) != 1:
        raise RuntimeError(f'Continuous usage patch requires its tested source; found {actual}')
    return source.replace(OLD, NEW)
