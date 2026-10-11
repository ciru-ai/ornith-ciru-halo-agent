"""Published serving settings with the qualified wide prefill policy."""
from ornith_g256 import launch
from pathlib import Path
import re

original = launch.make_settings
original_parser = launch.parser


def normalize_cache_args(args):
    """Separate compiled input contracts before any serving imports."""
    base = Path(args.cache_directory)
    while base.parent.name == 'runtime-modes' and (
            base.name == 'text-only' or re.fullmatch(r'vision-[1-9][0-9]*', base.name)):
        base = base.parent.parent
    mode = 'vision-' + str(args.max_images_per_prompt) if args.enable_images else 'text-only'
    args.cache_directory = base / 'runtime-modes' / mode
    return args


def parser():
    result = original_parser()
    original_parse_args = result.parse_args

    def parse_args(args=None, namespace=None):
        return normalize_cache_args(original_parse_args(args, namespace))

    result.parse_args = parse_args
    return result


def settings(args):
    normalize_cache_args(args)
    result = original(args)
    result.update(worker_cls='ornith_g256.iu4_correctness.worker.Worker',
                  max_num_batched_tokens=8192, long_prefill_token_threshold=4480)
    if args.cache_gib <= 16:
        result['gpu_memory_utilization'] = 0.35
    return result


launch.make_settings = settings
launch.parser = parser

if __name__ == '__main__':
    launch.main()
