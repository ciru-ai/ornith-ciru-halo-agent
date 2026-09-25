"""Published serving settings with the qualified wide prefill policy."""
from ornith_g256 import launch

original = launch.make_settings


def settings(args):
    result = original(args)
    result.update(worker_cls='ornith_g256.optimized.worker.Worker',
                  max_num_batched_tokens=8192, long_prefill_token_threshold=4480)
    return result


launch.make_settings = settings

if __name__ == '__main__':
    launch.main()
