__version__ = "2.3.2.post1"

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.modules.mamba2 import Mamba2
except Exception as exc:
    Mamba2 = None
    Mamba2_import_error = exc

try:
    from mamba_ssm.modules.mamba3 import Mamba3
except Exception as exc:
    Mamba3 = None
    Mamba3_import_error = exc

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except Exception as exc:
    MambaLMHeadModel = None
    MambaLMHeadModel_import_error = exc
