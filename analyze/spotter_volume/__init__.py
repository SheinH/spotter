"""spotter_volume: a small overlay that counts how long you aimed, and hides while you aim."""
import os

# dxcam converts captured pixels with OpenMP. By default OpenMP worker threads spin-wait after
# every call, which for 30 small grabs a second burned ~8 CPU cores doing nothing. Passive
# waiting makes them sleep. Must be set before dxcam (and its OpenMP runtime) is loaded.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
