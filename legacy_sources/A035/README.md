# Optimization prototypes (require lhd/ and st/code/ next to the scripts; opt1_sbi/opt3 need inv005.pkl, vis005.pkl from run_dns.py)
opt2_time.py      exponential (Magnus-midpoint) integrator vs implicit Euler, planar-shock kinematic test
opt1_anchored.py  current-Delaunay vs material connectivity, homogeneous shear+compression, exact Gaussian reference
opt1_sbi.py pe|visc  material connectivity on shock-bubble trajectories (inverted-triangle fraction, mixing)
opt3_thermo.py    position-only p and T (isentropic between hot-step blocks + Rankine-Hugoniot) vs DNS
