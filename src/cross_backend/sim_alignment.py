"""Load measured mapping and camera/scene provenance for aligned simulation."""
import json
from pathlib import Path
from .evaluation import digest
from .paired_backend_study import ALIGNMENT_COMPONENTS, read_reference
from .sim_policy_mapping import mapping_from_spec


def load_sim_alignment(path, backend):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    if spec.get('schema_version') != 1 or spec.get('scene_sha256') != backend.scene_sha256:
        raise ValueError('Alignment scene/mesh identity mismatch')
    calibration=spec.get('robot_calibration_sha256')
    if not isinstance(calibration,str) or len(calibration)!=64 or any(c not in '0123456789abcdef' for c in calibration):
        raise ValueError('Robot calibration identity missing')
    if not spec.get('reviewer') or not spec.get('reviewed_utc'):
        raise ValueError('Measured alignment review missing')
    for name in ALIGNMENT_COMPONENTS:
        component = spec.get(name, {})
        if component.get('status') != 'measured_and_reviewed' or not component.get('evidence'):
            raise ValueError('Unmeasured alignment component: ' + name)
        for reference in component['evidence']:
            read_reference(reference, path.parent, as_json=False)
    mapping = mapping_from_spec(spec)
    return mapping, {'alignment_spec_sha256': digest(path),
                     'scene_sha256': backend.scene_sha256,
                     'robot_calibration_sha256':calibration,
                     'physical_alignment_verified': True,
                     'camera_alignment_verified': True,
                     'alignment_review_basis': 'External measurements and reviewer; file integrity verified by loader'}
