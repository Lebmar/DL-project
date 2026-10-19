"""Slurm environment diagnostic on old validation-only arrays, never new tests."""
import hashlib
import json
import os
from pathlib import Path
import platform
import numpy as np
from train_stage2 import generate_paths, array_sha256

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID')
    reference=json.loads((ROOT/'outputs/neural_validation_reference/manifest.json').read_text())
    results=[]
    for row in reference['panels']:
        path=ROOT/row['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['file_sha256']
        with np.load(path,allow_pickle=False) as f: original=f['observations']
        trained=json.loads((ROOT/row['training_manifest']).read_text())
        assert array_sha256(original)==row['array_sha256']==trained['data_sha256']['validation']
        tc=trained['training_config'];linux=generate_paths(tc['validation_data'],tc['generator'])
        diff=np.abs(linux-original)
        results.append({'panel':row['panel'],'shape':list(original.shape),'original_array_sha256':array_sha256(original),
                        'linux_regenerated_sha256':array_sha256(linux),'different_float64_values':int(np.count_nonzero(diff)),
                        'maximum_absolute_error':float(diff.max()),
                        'maximum_relative_error':float((diff/np.maximum(np.abs(original),1e-300)).max()),
                        'model_float32_inputs_identical':bool(np.array_equal(linux.astype(np.float32),original.astype(np.float32)))})
    out=ROOT/'outputs/hpc_long_window_validation';out.mkdir(exist_ok=True)
    result={'purpose':'Platform portability diagnosis only; no new study data', 'slurm_job_id':os.environ['SLURM_JOB_ID'],
            'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),'panels':results}
    with (out/'portability.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))
