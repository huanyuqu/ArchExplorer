# GPU assembly of independent components

`gpu_assembly.*`, `gpu_transport.*` and `gpu_entry.cc` connect the public SM,
ICNT, L2 and DRAM factories through the generic Runtime. They own whole-GPU
phase wiring and trace-front-end orchestration. They do not provide another
copy of any component's mechanism.

These files are moved byte for byte. Configuration, program dispatch and
transaction services remain in `components/_support/accelsim/native`;
component factory code lives under each hardware component. The original
public flat headers, APIs and service lifetimes remain in force. The support
directory's `NOTICE` contains the shared license notices.
