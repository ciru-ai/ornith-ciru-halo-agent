"""The qualified streaming method must preserve length stops and frame errors."""
import asyncio
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'upstream'))


class FinishReason(unittest.TestCase):
    def test_complete_arguments_do_not_hide_length_stop(self):
        fixture=(HERE/'upstream/test_stream_usage.py').read_text()
        fixture=fixture.removesuffix('asyncio.run(main())\n')
        fixture=fixture.replace("finish_reason='stop' if finished else None", "finish_reason='length' if finished else None")
        ns={'__file__':str(HERE/'upstream/test_stream_usage.py')}
        exec(compile(fixture,'<length-fixture>','exec'),ns)
        shipped=(HERE.parents[1]/'plugin-site/ornith_g256/_vllm_correctness/chat_completion_serving_r03.py').read_text()
        method=ns['compile_method'](shipped)
        for continuous in (True,False):
            chunks,_=asyncio.run(ns['run'](method,continuous=continuous))
            finish=[choice['finish_reason'] for item in chunks if isinstance(item,dict)
                    for choice in item.get('choices',[]) if choice.get('finish_reason')]
            self.assertEqual(finish,['length'])
            failed,_=asyncio.run(ns['run'](method,continuous=continuous,invalid=True))
            self.assertTrue(any(isinstance(item,dict) and 'error' in item for item in failed))
            self.assertEqual(failed[-1],'[DONE]')


if __name__=='__main__':unittest.main()
