"""Finite source identities for read-only component migration.

Old maps below identify complete accepted producers. Path relocation is allowed
only for these exact vectors. A rehashed arbitrary producer source is refused;
regeneration still checks the complete graph, numerics, ISA and raw counters.
"""
from arch_explorer.components._support.paths import package_root
from pathlib import Path
import hashlib

PROGRAM_PATHS = {'components/accelsim.py': 'components/accelsim/declarations.py',
 'mapping/accelsim_program.py': 'components/accelsim/lowering.py',
 'simulator/accelsim/native_isa.py': 'components/accelsim/native_isa.py',
 'simulator/accelsim/program.py': 'components/accelsim/program.py',
 'simulator/accelsim/program_stats.py': 'components/accelsim/program_stats.py',
 'simulator/accelsim/runner.py': 'components/accelsim/runner.py',
 'simulator/accelsim/trace_stream.py': 'components/accelsim/trace_stream.py',
 'workload/full_models.py': 'workload/full_models.py',
 'workload/full_primitives.py': 'workload/full_primitives.py',
 'workload/operators.py': 'workload/operators.py',
 'workload/reference_execution.py': 'workload/reference_execution.py'}

LEGACY_PROGRAM_SOURCES = {'sim35': {'components/accelsim.py': '8e552f6df657801b0b59abac1a6f749c2a47756a4e419ad0f0660fd8926bab97',
           'mapping/accelsim_program.py': 'a0843ae6ff76d3abc35399f07131fbbd81e4bb538f42a730711c288dc50a3e76',
           'simulator/accelsim/native_isa.py': '0072ea7bdcc308f0c0d3d0b2a3b4261cfd6366743960840e8ffce97b3a5221aa',
           'simulator/accelsim/program.py': 'eb59dacc3cb18c2b6111b83972bd1eed77ec704519a8d624b2f948edaab4732f',
           'simulator/accelsim/program_stats.py': '84fbf90123dac59b457c5775ce6f372a1d44878f483cc4e9760793222af8fa8b',
           'simulator/accelsim/runner.py': 'bc04214d1c18ee9779131777e689119040e77aa9cc03bbf58a272f5c4a6bbcd2',
           'simulator/accelsim/trace_stream.py': 'a061e2b656ecd3b733fab3782c4a6130ab28aa15cbcd667d7b517c340d7b6c2f',
           'workload/full_models.py': 'd627187e47f5cffa3a1838deb7ef8b47b10d4896f9abbe1d5167d42cd465b46a',
           'workload/full_primitives.py': 'cfb00dd77dd6889fa49ecb5dd7ed2f9bba75b9588c87f945fc19d399d93f0488',
           'workload/operators.py': '1d1f8a01f85b978a015e76fc4b18843a4fe23c272ded2645a0ac49984b9e7d74',
           'workload/reference_execution.py': '23360b707ffaa238ad376f76365acf833045ab50527bc27f8be9a7a3dce2cace'},
 'sim35a': {'components/accelsim.py': '8e552f6df657801b0b59abac1a6f749c2a47756a4e419ad0f0660fd8926bab97',
            'mapping/accelsim_program.py': 'a0843ae6ff76d3abc35399f07131fbbd81e4bb538f42a730711c288dc50a3e76',
            'simulator/accelsim/native_isa.py': '0072ea7bdcc308f0c0d3d0b2a3b4261cfd6366743960840e8ffce97b3a5221aa',
            'simulator/accelsim/program.py': '97edd04cbb556497ca5f94f5650f80dc54a2135a907329012d54944a369fdaf9',
            'simulator/accelsim/program_stats.py': '84fbf90123dac59b457c5775ce6f372a1d44878f483cc4e9760793222af8fa8b',
            'simulator/accelsim/runner.py': 'bc04214d1c18ee9779131777e689119040e77aa9cc03bbf58a272f5c4a6bbcd2',
            'simulator/accelsim/trace_stream.py': 'a061e2b656ecd3b733fab3782c4a6130ab28aa15cbcd667d7b517c340d7b6c2f',
            'workload/full_models.py': 'd627187e47f5cffa3a1838deb7ef8b47b10d4896f9abbe1d5167d42cd465b46a',
            'workload/full_primitives.py': 'cfb00dd77dd6889fa49ecb5dd7ed2f9bba75b9588c87f945fc19d399d93f0488',
            'workload/operators.py': '1d1f8a01f85b978a015e76fc4b18843a4fe23c272ded2645a0ac49984b9e7d74',
            'workload/reference_execution.py': '23360b707ffaa238ad376f76365acf833045ab50527bc27f8be9a7a3dce2cace'}}

LEGACY_ADAPTER_SOURCES = {'sim28': {'archs/accelsim.py': '253c43b9bbf4db6916d9ecaebf17983aac53d7521d8d40f71e174101c63e693a',
           'components/accelsim.py': '783d9c56d193abc9f76bdfaede623d4de94d90e06fa22f61e3b4e56a4f7ea29e',
           'simulator/accelsim/__init__.py': 'b55d4c5f32f4d6052ce16ef53d7964b194cb41a8e0fbb379cf173a63add605fe',
           'simulator/accelsim/backend.py': 'b3db8cfc6d028a0e6c3520ddcdc89caeab9a353a2623f637e3a4d78a8946b878',
           'simulator/accelsim/capabilities.py': '9c07f9ea89ad4f0af92a8614d7b322f1a7180c4c4e11bd40e5cf84e761978a89',
           'simulator/accelsim/capture_input.py': '30166ebf5ad174e422d7e4f410e1e871148eed92a4b71cdea8e324da2e28ee5d',
           'simulator/accelsim/config.py': 'd24d5d0bcd83e809facd471408d16f75cebb3ef4a9d1320b80c1e1641028f95e',
           'simulator/accelsim/example.py': 'ff8739a870e1e64bc09c9022855b442e2ef19b575370f6e3ac26b66fbcf70dc0',
           'simulator/accelsim/isa.py': 'b432d90be2ad11101a4f37489596b50c34d751fa538a2b3de2888a377ef07824',
           'simulator/accelsim/native_isa.py': '0072ea7bdcc308f0c0d3d0b2a3b4261cfd6366743960840e8ffce97b3a5221aa',
           'simulator/accelsim/profile.py': '144f0e30818360ffc12f202a72a1554881b602465bf86afc78c16cec957d23d0',
           'simulator/accelsim/protocol.py': 'ec8b57316ba208b5787752acf030dea9944a82a6727962c2425e5f9a642bb912',
           'simulator/accelsim/runner.py': 'd263038df9eb68521cd2c9c9fe6fd42b16308572d43067ddf76637ba98be50bd',
           'simulator/accelsim/stats.py': 'a15d9e3009dbc820c485116309f5508fb01ee07af820ecb64ccdcc5d6ff0131b',
           'simulator/accelsim/trace.py': '76ab4e783af9e826a92d8c31ac5518f34b975751876dcc5386c45b7fdbf1efcc',
           'simulator/accelsim/trace_stream.py': 'a061e2b656ecd3b733fab3782c4a6130ab28aa15cbcd667d7b517c340d7b6c2f',
           'simulator/accelsim/trace_v2.py': '86fdf4b49fdcec6d2a6ba0d69716ffb5f97a4221eb3d4c6e4ae689688b280453',
           'simulator/accelsim/typed_trace.py': 'b5a79212aa41961831544d4f3eba024cd075075b808db2773a74fcdd8b1c7584',
           'simulator/accelsim/variant.py': '38b409b6de28ae9dac0851daa0aa7cd3f9c679aa4a7bda37fa2967eefb26e5a4'},
 'sim29': {'archs/accelsim.py': '7a81d8c9565c184dbc6e691ea5480fff9a57546df847d6f4e2cbec8bae14ae81',
           'components/accelsim.py': '8e552f6df657801b0b59abac1a6f749c2a47756a4e419ad0f0660fd8926bab97',
           'simulator/accelsim/__init__.py': 'b55d4c5f32f4d6052ce16ef53d7964b194cb41a8e0fbb379cf173a63add605fe',
           'simulator/accelsim/backend.py': '23eb7e02a0aa1a0466fd6bffc244f314bb5516b0ded936dabd08345343ebafcc',
           'simulator/accelsim/capabilities.py': '6fafff2c5baf5c945f85d50b8dd350717c63953a20e74b2f245379be9883dc58',
           'simulator/accelsim/capture_input.py': '30166ebf5ad174e422d7e4f410e1e871148eed92a4b71cdea8e324da2e28ee5d',
           'simulator/accelsim/config.py': '5018dd5f25520e1cffe4bff1721e6f2e5ce626d572c53d1e122e89a9aa6658b2',
           'simulator/accelsim/constraints.py': '416803cbbfb35b32c2593336861a3031a661b2496655cf784bcfb654187eb38f',
           'simulator/accelsim/device.py': '2230b11303258a3b8673e140f8e1d1679b329dc7fb37b06f5d23f452b34a6dc7',
           'simulator/accelsim/example.py': 'ff8739a870e1e64bc09c9022855b442e2ef19b575370f6e3ac26b66fbcf70dc0',
           'simulator/accelsim/isa.py': 'b432d90be2ad11101a4f37489596b50c34d751fa538a2b3de2888a377ef07824',
           'simulator/accelsim/native_isa.py': '0072ea7bdcc308f0c0d3d0b2a3b4261cfd6366743960840e8ffce97b3a5221aa',
           'simulator/accelsim/profile.py': '144f0e30818360ffc12f202a72a1554881b602465bf86afc78c16cec957d23d0',
           'simulator/accelsim/protocol.py': 'ec8b57316ba208b5787752acf030dea9944a82a6727962c2425e5f9a642bb912',
           'simulator/accelsim/runner.py': 'd263038df9eb68521cd2c9c9fe6fd42b16308572d43067ddf76637ba98be50bd',
           'simulator/accelsim/stats.py': 'a15d9e3009dbc820c485116309f5508fb01ee07af820ecb64ccdcc5d6ff0131b',
           'simulator/accelsim/trace.py': '76ab4e783af9e826a92d8c31ac5518f34b975751876dcc5386c45b7fdbf1efcc',
           'simulator/accelsim/trace_stream.py': 'a061e2b656ecd3b733fab3782c4a6130ab28aa15cbcd667d7b517c340d7b6c2f',
           'simulator/accelsim/trace_v2.py': '86fdf4b49fdcec6d2a6ba0d69716ffb5f97a4221eb3d4c6e4ae689688b280453',
           'simulator/accelsim/typed_trace.py': 'b5a79212aa41961831544d4f3eba024cd075075b808db2773a74fcdd8b1c7584',
           'simulator/accelsim/variant.py': '38b409b6de28ae9dac0851daa0aa7cd3f9c679aa4a7bda37fa2967eefb26e5a4'},
 'sim35-native-vit-32': {'archs/accelsim.py': '7a81d8c9565c184dbc6e691ea5480fff9a57546df847d6f4e2cbec8bae14ae81',
                         'components/accelsim.py': '8e552f6df657801b0b59abac1a6f749c2a47756a4e419ad0f0660fd8926bab97',
                         'simulator/accelsim/__init__.py': 'b55d4c5f32f4d6052ce16ef53d7964b194cb41a8e0fbb379cf173a63add605fe',
                         'simulator/accelsim/backend.py': 'ce1447cb87a8c3242d009772d87d9ebd367ba90278c5edbaf654ae8c1b49a1b5',
                         'simulator/accelsim/capabilities.py': '7bb80a00754e935a2c6a2fbd3f4528fb0944d0b099ebb4924f5e0c6c466bc45f',
                         'simulator/accelsim/capture_input.py': '0c5c899208183b8bc36f673416553156ca469934965277b4451c255b88b384fb',
                         'simulator/accelsim/config.py': 'ca1c57d6e2054654ce9f879b79e7ef1273b7b7663efb498fb34e60b3db44e80e',
                         'simulator/accelsim/constraints.py': '416803cbbfb35b32c2593336861a3031a661b2496655cf784bcfb654187eb38f',
                         'simulator/accelsim/device.py': '23ae314bd6424715ef934ee5593485dd93d9c4eb85327f9a2d272953b9819294',
                         'simulator/accelsim/example.py': 'ff8739a870e1e64bc09c9022855b442e2ef19b575370f6e3ac26b66fbcf70dc0',
                         'simulator/accelsim/isa.py': 'b432d90be2ad11101a4f37489596b50c34d751fa538a2b3de2888a377ef07824',
                         'simulator/accelsim/measurement_state.py': '2077098971c81378880508be34ecdf387c32718c723d92b8628d7735e36601d7',
                         'simulator/accelsim/microbundle.py': '3179064da904683a4124ef2cc094e1822eec9a73046f68928415b15de236e5b8',
                         'simulator/accelsim/microcampaign.py': 'ccb056824a2f650336ddf9298d809c3b4a7cd13dd683ef257ba339573761174c',
                         'simulator/accelsim/microcapture.py': 'd89e7e8535c71cb91b689420cf6f04499e1c58aafa3cb4b1111244bb2a10948d',
                         'simulator/accelsim/microcatalog.py': '7796e97a0e5d5ec68d3a4bcedac71fb6776728225f90d6ceeaf482cd0b2129d3',
                         'simulator/accelsim/microclosure.py': 'c47314afc9a8226b9de22683b89454d99685433a07fc48d2d6bb03a864d14e32',
                         'simulator/accelsim/microcomparison.py': '44fb5ad66f3f58d88c31314aae578cbd9b325b9e924920f4033abed092cc2208',
                         'simulator/accelsim/microcontract.py': 'e674420f4130f5557016e3e323ca20981c5ee090c3ed2e047ddeb4285fe61b85',
                         'simulator/accelsim/microexperiment.py': 'b0e3655da93f68adcc450a5ad46b6f8783968bde3dd6b552d4669d55863979af',
                         'simulator/accelsim/microguard.py': '9909a301b8897489a9c6bff871e130ef160e673fb308ac567a807a5da8fe1471',
                         'simulator/accelsim/microrecords.py': 'd3c87ad3ad6f44f4735e70f33eb0118df1d9a786921af3b8d4ab4b9a7b450b26',
                         'simulator/accelsim/microrehearsal.py': 'd6a88a89138b99548011e4a838e3913c347d2afd2fb222fe3ddeb6fd1b382dfb',
                         'simulator/accelsim/microstate.py': 'f05e1eb0da83a753721ea543ab9925ba4a3778525784b78deaa2c6fef9712b5b',
                         'simulator/accelsim/microworker.py': '5a17e10ee6ffe3688315fb1c3007475280dcec0765006bc2edb5904868d6c528',
                         'simulator/accelsim/native_isa.py': '0072ea7bdcc308f0c0d3d0b2a3b4261cfd6366743960840e8ffce97b3a5221aa',
                         'simulator/accelsim/profile.py': '144f0e30818360ffc12f202a72a1554881b602465bf86afc78c16cec957d23d0',
                         'simulator/accelsim/program.py': 'eb59dacc3cb18c2b6111b83972bd1eed77ec704519a8d624b2f948edaab4732f',
                         'simulator/accelsim/program_stats.py': '84fbf90123dac59b457c5775ce6f372a1d44878f483cc4e9760793222af8fa8b',
                         'simulator/accelsim/protocol.py': 'bc64329cea3bdb0c733adfe40b1789ff24a587db775a7fa4618023d533d6f6db',
                         'simulator/accelsim/result_contract.py': '51666b2ea974830e197793fd2330c1e772f52227c43966c1e520a2eea9ca7303',
                         'simulator/accelsim/runner.py': 'bc04214d1c18ee9779131777e689119040e77aa9cc03bbf58a272f5c4a6bbcd2',
                         'simulator/accelsim/state_device.py': '831bd37999d51f9a0e199ffe0e1ad466e5b91498acdbbd9f4c33ae494104f1b3',
                         'simulator/accelsim/state_device_reader.py': '4fa5de235d0b758ebd06668f4f27bda333b1c9d3dd576eed81bece78ed600fa1',
                         'simulator/accelsim/state_fixture.py': '9eae61e4ba0ec3af80bbec6db56bd404f9f50ccf698101a0642b6f5e407e5c38',
                         'simulator/accelsim/state_lifecycle.py': '12a23efc5bc5bdb4624b9616433f1c985c7e5fec3addad2182455a0ed95f6cb7',
                         'simulator/accelsim/state_observation.py': '52d1d15d125d89fcecc4a489c527c19e023c98f321e539397a42f1fd5679b09c',
                         'simulator/accelsim/state_resource.py': '2aa0022ba460d8e1c38b829270dbca3d488ab079aa7192cc5c0d253312a12538',
                         'simulator/accelsim/stats.py': 'a15d9e3009dbc820c485116309f5508fb01ee07af820ecb64ccdcc5d6ff0131b',
                         'simulator/accelsim/trace.py': '76ab4e783af9e826a92d8c31ac5518f34b975751876dcc5386c45b7fdbf1efcc',
                         'simulator/accelsim/trace_stream.py': 'a061e2b656ecd3b733fab3782c4a6130ab28aa15cbcd667d7b517c340d7b6c2f',
                         'simulator/accelsim/trace_v2.py': '86fdf4b49fdcec6d2a6ba0d69716ffb5f97a4221eb3d4c6e4ae689688b280453',
                         'simulator/accelsim/typed_trace.py': 'b5a79212aa41961831544d4f3eba024cd075075b808db2773a74fcdd8b1c7584',
                         'simulator/accelsim/variant.py': '38b409b6de28ae9dac0851daa0aa7cd3f9c679aa4a7bda37fa2967eefb26e5a4'},
 'sim36-baseline': {'archs/accelsim.py': '7a81d8c9565c184dbc6e691ea5480fff9a57546df847d6f4e2cbec8bae14ae81',
                    'components/accelsim.py': '8e552f6df657801b0b59abac1a6f749c2a47756a4e419ad0f0660fd8926bab97',
                    'simulator/accelsim/__init__.py': 'b55d4c5f32f4d6052ce16ef53d7964b194cb41a8e0fbb379cf173a63add605fe',
                    'simulator/accelsim/backend.py': 'ce1447cb87a8c3242d009772d87d9ebd367ba90278c5edbaf654ae8c1b49a1b5',
                    'simulator/accelsim/capabilities.py': '7bb80a00754e935a2c6a2fbd3f4528fb0944d0b099ebb4924f5e0c6c466bc45f',
                    'simulator/accelsim/capture_input.py': '0c5c899208183b8bc36f673416553156ca469934965277b4451c255b88b384fb',
                    'simulator/accelsim/config.py': 'ca1c57d6e2054654ce9f879b79e7ef1273b7b7663efb498fb34e60b3db44e80e',
                    'simulator/accelsim/constraints.py': '416803cbbfb35b32c2593336861a3031a661b2496655cf784bcfb654187eb38f',
                    'simulator/accelsim/device.py': '23ae314bd6424715ef934ee5593485dd93d9c4eb85327f9a2d272953b9819294',
                    'simulator/accelsim/example.py': 'ff8739a870e1e64bc09c9022855b442e2ef19b575370f6e3ac26b66fbcf70dc0',
                    'simulator/accelsim/isa.py': 'b432d90be2ad11101a4f37489596b50c34d751fa538a2b3de2888a377ef07824',
                    'simulator/accelsim/measurement_state.py': '2077098971c81378880508be34ecdf387c32718c723d92b8628d7735e36601d7',
                    'simulator/accelsim/microbundle.py': '3179064da904683a4124ef2cc094e1822eec9a73046f68928415b15de236e5b8',
                    'simulator/accelsim/microcampaign.py': '8eb82eaba668e15b0be7aa0a86cc72375ccca7c43a89e33044f456eedb2d64be',
                    'simulator/accelsim/microcapture.py': 'd89e7e8535c71cb91b689420cf6f04499e1c58aafa3cb4b1111244bb2a10948d',
                    'simulator/accelsim/microcatalog.py': '7796e97a0e5d5ec68d3a4bcedac71fb6776728225f90d6ceeaf482cd0b2129d3',
                    'simulator/accelsim/microclosure.py': 'c47314afc9a8226b9de22683b89454d99685433a07fc48d2d6bb03a864d14e32',
                    'simulator/accelsim/microcomparison.py': '44fb5ad66f3f58d88c31314aae578cbd9b325b9e924920f4033abed092cc2208',
                    'simulator/accelsim/microcontract.py': 'e674420f4130f5557016e3e323ca20981c5ee090c3ed2e047ddeb4285fe61b85',
                    'simulator/accelsim/microexperiment.py': 'b0e3655da93f68adcc450a5ad46b6f8783968bde3dd6b552d4669d55863979af',
                    'simulator/accelsim/microguard.py': '9909a301b8897489a9c6bff871e130ef160e673fb308ac567a807a5da8fe1471',
                    'simulator/accelsim/microrecords.py': '3912ab19fd82c20c244e68f0553772baaaa5473e40d75214a7db1eacf1e70291',
                    'simulator/accelsim/microrehearsal.py': 'd6a88a89138b99548011e4a838e3913c347d2afd2fb222fe3ddeb6fd1b382dfb',
                    'simulator/accelsim/microstate.py': 'f05e1eb0da83a753721ea543ab9925ba4a3778525784b78deaa2c6fef9712b5b',
                    'simulator/accelsim/microworker.py': '5a17e10ee6ffe3688315fb1c3007475280dcec0765006bc2edb5904868d6c528',
                    'simulator/accelsim/native_isa.py': '0072ea7bdcc308f0c0d3d0b2a3b4261cfd6366743960840e8ffce97b3a5221aa',
                    'simulator/accelsim/profile.py': '144f0e30818360ffc12f202a72a1554881b602465bf86afc78c16cec957d23d0',
                    'simulator/accelsim/program.py': '97edd04cbb556497ca5f94f5650f80dc54a2135a907329012d54944a369fdaf9',
                    'simulator/accelsim/program_stats.py': '84fbf90123dac59b457c5775ce6f372a1d44878f483cc4e9760793222af8fa8b',
                    'simulator/accelsim/protocol.py': 'bc64329cea3bdb0c733adfe40b1789ff24a587db775a7fa4618023d533d6f6db',
                    'simulator/accelsim/result_contract.py': '51666b2ea974830e197793fd2330c1e772f52227c43966c1e520a2eea9ca7303',
                    'simulator/accelsim/runner.py': 'bc04214d1c18ee9779131777e689119040e77aa9cc03bbf58a272f5c4a6bbcd2',
                    'simulator/accelsim/state_device.py': '7e93a90b934c8b9fa9260e44e5546b68384384721daa7a45bd38237213ee248a',
                    'simulator/accelsim/state_device_reader.py': '3aa5d0fe4e2b1904c6c4df6ff563b23387daf431db26497651b3b757fc3f1d28',
                    'simulator/accelsim/state_fixture.py': '9eae61e4ba0ec3af80bbec6db56bd404f9f50ccf698101a0642b6f5e407e5c38',
                    'simulator/accelsim/state_idle.py': 'ef117cd3318d1493e55fd50745f6c5e9aa63b952748d9adafb0af7c1ed6a0c94',
                    'simulator/accelsim/state_lifecycle.py': '12a23efc5bc5bdb4624b9616433f1c985c7e5fec3addad2182455a0ed95f6cb7',
                    'simulator/accelsim/state_observation.py': '52d1d15d125d89fcecc4a489c527c19e023c98f321e539397a42f1fd5679b09c',
                    'simulator/accelsim/state_resource.py': 'fd49d263d8f37ef61f877894ea2f2cf976d3f16d210da4f7420a4169bb2b94ff',
                    'simulator/accelsim/stats.py': 'a15d9e3009dbc820c485116309f5508fb01ee07af820ecb64ccdcc5d6ff0131b',
                    'simulator/accelsim/trace.py': '76ab4e783af9e826a92d8c31ac5518f34b975751876dcc5386c45b7fdbf1efcc',
                    'simulator/accelsim/trace_stream.py': 'a061e2b656ecd3b733fab3782c4a6130ab28aa15cbcd667d7b517c340d7b6c2f',
                    'simulator/accelsim/trace_v2.py': '86fdf4b49fdcec6d2a6ba0d69716ffb5f97a4221eb3d4c6e4ae689688b280453',
                    'simulator/accelsim/typed_trace.py': 'b5a79212aa41961831544d4f3eba024cd075075b808db2773a74fcdd8b1c7584',
                    'simulator/accelsim/variant.py': '38b409b6de28ae9dac0851daa0aa7cd3f9c679aa4a7bda37fa2967eefb26e5a4'}}

def workload_source_paths():
    """Close over the workload dispatch/validation/lowering implementation."""
    root = package_root()
    return tuple(p.relative_to(root).as_posix()
                 for p in sorted((root / 'workload').rglob('*.py')))


def program_source_era(producer, consumer):
    if not isinstance(producer, dict) or not isinstance(consumer, dict):
        raise ValueError('complete program source binding required')
    from arch_explorer.components.accelsim.program_sources import sources as current_sources
    expected = set(current_sources())
    if {k for k in consumer if not k.startswith('extension:')} != expected:
        raise ValueError('consumer program source closure differs')
    if producer == consumer:
        return 'current'
    from .layout_history import source_era
    try:
        return source_era(producer, consumer, 'program', read_only=True)
    except ValueError:
        pass
    for era, frozen in LEGACY_PROGRAM_SOURCES.items():
        if producer == frozen:
            # Model/reference semantics did not move and must remain byte exact.
            if any(consumer[new] != frozen[old] for old, new in PROGRAM_PATHS.items() if old.startswith('workload/')):
                raise ValueError('legacy workload semantics no longer match this migration')
            return era
    raise ValueError('program source binding differs without an exact finite migration')


def adapter_sources():
    from arch_explorer.components.accelsim.program_sources import sources
    return {k:v for k,v in sources().items() if k.startswith('components/')}


def adapter_source_era(sources):
    if not isinstance(sources, dict):
        raise ValueError('complete native adapter source identity required')
    if sources == adapter_sources():
        return 'current'
    from .layout_history import frozen
    if sources == frozen()['adapter']:
        return 'sim40-component-layout-v1-read-only'
    for era, frozen in LEGACY_ADAPTER_SOURCES.items():
        if sources == frozen:
            return era
    raise ValueError('native adapter source vector has no exact registered producer')


BASELINE_RESULT_V2_SOURCES = {'components/accelsim.py': '8e552f6df657801b0b59abac1a6f749c2a47756a4e419ad0f0660fd8926bab97',
 'simulator/accelsim/backend.py': 'ce1447cb87a8c3242d009772d87d9ebd367ba90278c5edbaf654ae8c1b49a1b5',
 'simulator/accelsim/config.py': 'ca1c57d6e2054654ce9f879b79e7ef1273b7b7663efb498fb34e60b3db44e80e',
 'simulator/accelsim/constraints.py': '416803cbbfb35b32c2593336861a3031a661b2496655cf784bcfb654187eb38f',
 'simulator/accelsim/device.py': '23ae314bd6424715ef934ee5593485dd93d9c4eb85327f9a2d272953b9819294',
 'simulator/accelsim/profile.py': '144f0e30818360ffc12f202a72a1554881b602465bf86afc78c16cec957d23d0',
 'simulator/accelsim/result_contract.py': '51666b2ea974830e197793fd2330c1e772f52227c43966c1e520a2eea9ca7303'}
