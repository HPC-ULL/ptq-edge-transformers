"""Energy measurement, per-device implementation + common contract
(technical_architecture.md Section 6.4/6.4.1).

`orin_power.py` is the Jetson-specific sensor implementation (INA3221,
GPU+SoC and CPU+CV rails) -- moved here verbatim from the project root,
internal logic untouched. `jetson_powermeter.py` is the Section 6.4 common-
contract wrapper around it: translates orin_power's own field names to the
project's homogeneous schema (`energy_total_j` -> `energy_raw_j`,
`energy_net_j` kept, `avg_power_w_raw`/`avg_power_w_net` derived), combines
that with the wrapped runner's own parsed performance output, and writes
ONE row per (run_id, device_id, repeat_id) to results/raw/ -- see that
module's docstring.

Section 6.4 also calls for a hardware-agnostic "dummy"/reference interface
module in this package (the contract's input/output shape, no real sensor)
so a future non-Jetson implementation (AccelPowerCape/INA219) has a
template to match -- not yet built here, only jetson_powermeter.py.
"""
