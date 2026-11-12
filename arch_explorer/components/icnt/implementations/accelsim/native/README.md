# Accel-Sim interconnect factory

`icnt.cc` owns `make_icnt`, its per-instance BookSim/Local state and public
transport methods. `icnt_api.hpp` keeps the original flat include name.
Directory migration preserves both files byte for byte. The existing ICNT
generator still extracts the fixed native mechanisms and records their exact
source closure.

Shared transaction/services headers and upstream license notices are under
`components/_support/accelsim/native`; GPU endpoint wiring is a separate
assembly. Discovery of these resources does not create a network.
