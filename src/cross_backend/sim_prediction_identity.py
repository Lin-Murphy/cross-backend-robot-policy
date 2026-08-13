"""File and observation identity checks for the offline simulation model worker."""
from pathlib import Path
import math

from .move_pot_policy import CAMERAS, JOINTS, UNITS, file_sha256


def _hash(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def verify_worker_request(request, *, trial_id, output_dir):
    if request.get('trial_id')!=trial_id:
        raise ValueError('worker trial identity mismatch')
    index=request.get('index')
    if type(index) is not int or index<0:
        raise ValueError('worker prediction index invalid')
    stamp=request.get('capture_sim_ns')
    observation_id=request.get('observation_id')
    if type(stamp) is not int or stamp<0 or not isinstance(observation_id,str) or not observation_id.startswith(trial_id+':frame:'):
        raise ValueError('worker observation identity invalid')
    expected=(Path(output_dir)/f'input-{index:03d}.npz').resolve()
    if Path(request.get('input','')).resolve()!=expected or not expected.is_file():
        raise ValueError('worker input path mismatch')
    expected_hash=request.get('input_sha256')
    if not _hash(expected_hash) or file_sha256(expected)!=expected_hash:
        raise ValueError('worker input hash mismatch')
    return expected


def verify_prediction_response(answer, metadata, *, trial_id, observation_id, capture_sim_ns,
                               prediction_index, input_sha256, model_sha256, output_dir):
    expected=(Path(output_dir)/f'prediction-{prediction_index:03d}.npz').resolve()
    if Path(answer.get('actions','')).resolve()!=expected or not expected.is_file():
        raise ValueError('prediction action path mismatch')
    if metadata.get('trial_id')!=trial_id or metadata.get('input_sha256')!=input_sha256:
        raise ValueError('prediction trial/input identity mismatch')
    if not _hash(metadata.get('actions_sha256')) or file_sha256(expected)!=metadata['actions_sha256']:
        raise ValueError('prediction actions hash mismatch')
    source=metadata.get('source')
    if not isinstance(source,dict) or source.get('observation_id')!=observation_id:
        raise ValueError('prediction observation identity mismatch')
    if source.get('state_capture_ns')!=capture_sim_ns or source.get('camera_capture_ns')!={k:capture_sim_ns for k in CAMERAS}:
        raise ValueError('prediction capture time mismatch')
    if source.get('model_sha256')!=model_sha256 or not _hash(model_sha256):
        raise ValueError('prediction checkpoint identity mismatch')
    if source.get('joint_names')!=list(JOINTS) or source.get('joint_units')!=list(UNITS):
        raise ValueError('prediction joint contract mismatch')
    start,end=metadata.get('prediction_host_start_ns'),metadata.get('prediction_host_end_ns')
    if type(start) is not int or type(end) is not int or start<0 or end<start:
        raise ValueError('prediction host timing invalid')
    if any(not isinstance(metadata.get(k),(int,float)) or not math.isfinite(metadata[k]) or metadata[k]<0
           for k in ('forward_ms','processing_ms')):
        raise ValueError('prediction cost invalid')
    return expected
