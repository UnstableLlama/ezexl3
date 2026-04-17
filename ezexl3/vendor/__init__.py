"""Install compatibility shims before any vendored script imports exllamav3."""
from ezexl3._exllamav3_compat import install_progress_shim as _install
_install()
