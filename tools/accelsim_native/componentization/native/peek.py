"""Read-only ICNT head queries, preserving each native pop arbitration order.

Original Pop/Advance implementations are unchanged. BookSim's existing
TopPacket loops on the first tail flag for a multi-flit packet; the adapter
therefore uses the known-complete packet's first data pointer directly. Native
PopPacket already asserts that every flit through its tail has that pointer.
"""
import hashlib

EXPECTED={
 'intersim2/interconnect_interface.cpp':'644dc5e2d5d0868cc4b16bbfba56ed4046240969a4b91a1daf80c5f9683b5233',
 'intersim2/interconnect_interface.hpp':'fd847448ef97bdeb1205fe340c7ab717c7462b36bb521d79c6e42f2f57d1dbc3',
 'gpgpu-sim/local_interconnect.cc':'02146e90be68286ba1620fda8f4079c59cae2df5160fc840b3181e5075a3cbaf',
 'gpgpu-sim/local_interconnect.h':'a67a6b02c1e236d3bc36100ac729d45f4e2bdaeec0db10a7a42244505d449d1a',
 'gpgpu-sim/icnt_wrapper.cc':'dde69b640c9b34e1b0ab1c5ec20a3cdbb69065311878f35f80baabc6cb449ad5',
 'gpgpu-sim/delayqueue.h':'804128a400d925d06bdd10aa2e1961f308f47f52dec5e1ca1bca9351fd9cdd4d',
}


def patch_peek(src):
    files={}
    for name,sha in EXPECTED.items():
        data=(src/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=sha: raise ValueError('ICNT peek input mismatch: '+name)
        files[name]=data.decode()
    def replace(name,old,new,count=1):
        if files[name].count(old)!=count: raise ValueError('ICNT peek anchor: '+name)
        files[name]=files[name].replace(old,new)
    h='intersim2/interconnect_interface.hpp'
    replace(h,'  virtual void* Pop(unsigned ouput_deviceID);',
            '  virtual void* Pop(unsigned ouput_deviceID);\n  void* AePeek(unsigned deviceID) const;')
    replace(h,'    void* TopPacket() const;',
            '    void* TopPacket() const;\n    void* AePeek() const { return _packet_n ? _buffer.front() : NULL; }')
    files['intersim2/interconnect_interface.cpp']+='''
// Side-effect-free head query with exactly Pop's virtual-channel turn.
void* InterconnectInterface::AePeek(unsigned deviceID) const {
  const int id = _node_map.at(deviceID);
  const int subnet = deviceID < _n_shader ? 1 : 0;
  int turn = _round_robin_turn[subnet][id];
  for (int vc=0; vc<_vcs; ++vc) {
    if (_boundary_buffer[subnet][id][turn].HasPacket())
      return _boundary_buffer[subnet][id][turn].AePeek();
    if (++turn == _vcs) turn=0;
  }
  return NULL;
}
'''
    h='gpgpu-sim/local_interconnect.h'
    replace(h,'  void* Pop(unsigned ouput_deviceID);',
            '  void* Pop(unsigned ouput_deviceID);\n  void* AePeek(unsigned deviceID) const;',2)
    replace(h,'  bool HasBuffer(unsigned deviceID, unsigned int size) const;',
            '  bool HasBuffer(unsigned deviceID, unsigned int size) const;\n  bool AeHasBuffer(unsigned deviceID, unsigned int size) const;')
    files['gpgpu-sim/local_interconnect.cc']+='''
void* xbar_router::AePeek(unsigned deviceID) const {
  assert(deviceID < total_nodes);
  return out_buffers[deviceID].empty() ? NULL : out_buffers[deviceID].front().data;
}
void* LocalInterconnect::AePeek(unsigned deviceID) const {
  return net[deviceID < n_shader ? 1 : 0]->AePeek(deviceID);
}
bool LocalInterconnect::AeHasBuffer(unsigned deviceID, unsigned int size) const {
  const int subnet=(n_subnets > 1 && deviceID >= n_shader) ? REPLY_NET : REQ_NET;
  return net[subnet]->Has_Buffer_In(deviceID,1,false);
}
'''
    files['gpgpu-sim/icnt_wrapper.cc']+='''
namespace ae_accelsim {
void* peek_icnt(unsigned output) {
  if (g_network_mode == 1) return g_icnt_interface->AePeek(output);
  if (g_network_mode == 2) return g_localicnt_interface->AePeek(output);
  throw std::runtime_error("SIM38: unsupported native ICNT mode");
}
bool probe_icnt(unsigned input, unsigned size) {
  if (g_network_mode == 1) return g_icnt_interface->HasBuffer(input,size);
  if (g_network_mode == 2) return g_localicnt_interface->AeHasBuffer(input,size);
  throw std::runtime_error("SIM38: unsupported native ICNT mode");
}
}
'''
    files['gpgpu-sim/icnt_wrapper.cc']='#include <stdexcept>\n'+files['gpgpu-sim/icnt_wrapper.cc']
    replace('gpgpu-sim/delayqueue.h','  bool empty() const { return m_head == NULL; }', '''  bool empty() const { return m_head == NULL; }
  // Drain observation ignores timing-only NULL slots; no FIFO mutation.
  bool ae_has_payload() const {
    for (auto* node=m_head; node; node=node->m_next)
      if (node->m_data) return true;
    return false;
  }''')
    for name,text in files.items(): (src/name).write_text(text)
    return {str(name):hashlib.sha256(text.encode()).hexdigest() for name,text in files.items()}
