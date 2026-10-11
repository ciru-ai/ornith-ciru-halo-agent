"""Private release repair: identify every executable plugin and native artifact."""
import hashlib
import json
from pathlib import Path


def identify_bundle(bundle):
    bundle = Path(bundle).resolve()
    records = []
    # Checkpoint weights do not alter installed monkeypatches. The serving
    # configuration already records the model path; never scan weight shards.
    for base in (bundle/'plugin-site', bundle/'native', bundle/'packaging'):
        for path in sorted(base.rglob('*')):
            if (path.is_file() and '__pycache__' not in path.parts
                    and path.suffix in ('.py', '.so', '.sh', '.json', '.safetensors')):
                records.append((str(path.relative_to(bundle)), hashlib.sha256(path.read_bytes()).hexdigest()))
    for name in ('serve.sh', 'generation-defaults.json', 'runtime-manifest.json'):
        path = bundle/name
        if path.is_file():
            records.append((name, hashlib.sha256(path.read_bytes()).hexdigest()))
    if not records:
        raise ValueError('No executable bundle files found')
    return hashlib.sha256(json.dumps(sorted(records), separators=(',', ':')).encode()).hexdigest()


if __name__ == '__main__':
    import sys
    print(identify_bundle(sys.argv[1]))
