"""CPU regression tests using the real public parser/environment/main bodies."""
import argparse
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
STAGE = Path(os.environ.get('CACHE_LAUNCH_TEST_STAGE', ROOT))


def load_launcher(stage):
    plugin = stage/'plugin-site/ornith_g256'
    tree = ast.parse((plugin/'launch.py').read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('parser', 'environment', 'main')]
    module = types.ModuleType('ornith_g256.launch')
    module.__dict__.update(argparse=argparse, json=json, os=os, Path=Path)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(plugin/'launch.py'), 'exec'), module.__dict__)
    original_calls = []
    def native_settings(args):
        original_calls.append(args.cache_directory)
        return dict(retained_native_setting=17, gpu_memory_utilization=0.8)
    module.make_settings = native_settings
    package = types.ModuleType('ornith_g256'); package.launch = module
    spec = importlib.util.spec_from_file_location('cache_mode_test_launcher', plugin/'optimized/launch.py')
    optimized = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'ornith_g256':package, 'ornith_g256.launch':module}):
        spec.loader.exec_module(optimized)
    return optimized, module, original_calls


class CacheModeTests(unittest.TestCase):
    def setUp(self):
        self.optimized, self.launch, self.calls = load_launcher(STAGE)
        self.common = ['--model','/model','--native-library-directory','/native','--cache-directory','/cache/build']

    def args(self, extra=()):
        return self.launch.parser().parse_args(self.common+list(extra))

    def test_parser_scopes_before_wrapper_environment(self):
        args = self.args(['--enable-images'])
        self.assertEqual(args.cache_directory, Path('/cache/build/runtime-modes/vision-1'))
        env = self.launch.environment(args.cache_directory)
        for key, suffix in [('XDG_CACHE_HOME',''), ('ORNITH_OPTIMIZED_CACHE',''),
                            ('AITER_JIT_DIR','/aiter'),('TRITON_CACHE_DIR','/triton'),
                            ('TORCHINDUCTOR_CACHE_DIR','/inductor'),('VLLM_CACHE_ROOT','/vllm')]:
            self.assertEqual(env[key], str(args.cache_directory)+suffix)

    def test_default_text_only(self):
        self.assertEqual(self.args().cache_directory, Path('/cache/build/runtime-modes/text-only'))

    def test_native_flag_precedence(self):
        self.assertEqual(self.args(['--enable-images','--text-only']).cache_directory.name, 'text-only')
        self.assertEqual(self.args(['--text-only','--enable-images']).cache_directory.name, 'vision-1')

    def test_image_limit_isolated(self):
        one=self.args(['--enable-images']); two=self.args(['--enable-images','--max-images-per-prompt','2'])
        self.assertNotEqual(one.cache_directory, two.cache_directory)
        self.assertEqual(two.cache_directory.name, 'vision-2')

    def test_settings_idempotent_and_native_body_preserved(self):
        args = self.args(['--enable-images']); expected=args.cache_directory
        result=self.optimized.settings(args); self.optimized.settings(args)
        self.assertEqual(args.cache_directory, expected)
        self.assertEqual(self.calls, [expected, expected])
        self.assertEqual(result, dict(retained_native_setting=17, gpu_memory_utilization=0.8,
            worker_cls='ornith_g256.iu4_correctness.worker.Worker', max_num_batched_tokens=8192,
            long_prefill_token_threshold=4480))
        args.cache_gib=16
        self.assertEqual(self.optimized.settings(args)['gpu_memory_utilization'], 0.35)

    def test_constructed_namespace_and_mode_change(self):
        args=argparse.Namespace(cache_directory=Path('/cache/build'),enable_images=True,max_images_per_prompt=3,cache_gib=32)
        self.optimized.settings(args)
        self.assertEqual(args.cache_directory, Path('/cache/build/runtime-modes/vision-3'))
        args.enable_images=False; self.optimized.settings(args)
        self.assertEqual(args.cache_directory, Path('/cache/build/runtime-modes/text-only'))

    def test_nested_valid_suffix_collapses(self):
        args=self.args(); args.cache_directory=Path('/cache/build/runtime-modes/vision-1/runtime-modes/text-only')
        self.optimized.settings(args)
        self.assertEqual(args.cache_directory, Path('/cache/build/runtime-modes/text-only'))

    def test_unrecognized_suffix_preserved(self):
        args=self.args(); args.cache_directory=Path('/cache/runtime-modes/vision-0')
        self.optimized.settings(args)
        self.assertEqual(args.cache_directory,Path('/cache/runtime-modes/vision-0/runtime-modes/text-only'))

    def test_original_parser_namespace_forwarded(self):
        namespace=argparse.Namespace(extra='retained')
        result=self.launch.parser().parse_args(self.common+['--enable-images'], namespace=namespace)
        self.assertIs(result, namespace); self.assertEqual(result.extra,'retained')
        self.assertEqual(result.cache_directory.name, 'vision-1')

    def test_actual_public_main_dry_run_uses_scoped_environment(self):
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            self.launch.main(self.common+['--enable-images','--max-images-per-prompt','2','--dry-run'])
        value=json.loads(out.getvalue())
        self.assertEqual(value['environment']['VLLM_CACHE_ROOT'],'/cache/build/runtime-modes/vision-2/vllm')
        self.assertEqual(self.calls, [Path('/cache/build/runtime-modes/vision-2')])


if __name__ == '__main__':
    unittest.main()
