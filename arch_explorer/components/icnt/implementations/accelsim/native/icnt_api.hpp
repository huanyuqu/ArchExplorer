// SPDX-License-Identifier: Apache-2.0
#ifndef AE_ACCELSIM_STANDALONE_ICNT_API_HPP
#define AE_ACCELSIM_STANDALONE_ICNT_API_HPP

#include "runtime.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ae_accelsim { namespace standalone {

enum class IcntPacketType { ReadRequest, WriteRequest, ReadReply, WriteReply };
struct LocalIcntConfig {
  unsigned input_buffer = 64, output_buffer = 64;
  unsigned arbitration = 1, grant_cycles = 1;
};
struct IcntConfig {
  unsigned mode = 1;
  unsigned request_nodes = 0, memory_nodes = 0;
  std::string booksim_config;
  LocalIcntConfig local;
};
struct IcntRouterMetrics {
  std::uint64_t cycles = 0, conflicts = 0, conflicts_util = 0;
  std::uint64_t cycles_util = 0, reqs_util = 0;
  std::uint64_t out_buffer_full = 0, out_buffer_util = 0;
  std::uint64_t in_buffer_full = 0, in_buffer_util = 0, packets = 0;
};
struct IcntMetrics {
  unsigned mode = 0, flit_bytes = 0;
  std::uint64_t cycles = 0, accepted = 0, delivered = 0, transfer_bytes = 0;
  std::uint64_t pending_packets = 0, live_flits = 0, live_credits = 0;
  std::uint64_t allocated_flits = 0, allocated_credits = 0;
  bool native_busy = false;
  std::vector<IcntRouterMetrics> local_routers;
};

// No execution/GPU service is required. Each object owns both directional
// networks, their native clocks/arbiters/configuration and all packet storage.
class Icnt {
 public:
  virtual ~Icnt() = default;
  virtual bool has_buffer(unsigned input, std::uint64_t transfer_bytes) const = 0;
  virtual bool push(unsigned input, unsigned output,
                    const ae_dynamic::Transaction&, IcntPacketType) = 0;
  virtual const ae_dynamic::Transaction* peek(unsigned output) const = 0;
  virtual bool pop(unsigned output, ae_dynamic::Transaction& result) = 0;
  // Same explicit interval boundary as native icnt_init: BookSim resets its
  // local time/statistics, LocalInterconnect's original Init is a no-op. This
  // does not clear queues, destroy packets or reset transport lifetime totals.
  virtual void begin_interval() = 0;
  virtual void advance() = 0;
  virtual bool busy() const = 0;
  virtual IcntMetrics metrics() const = 0;
};
std::unique_ptr<Icnt> make_icnt(const IcntConfig&);

}}  // namespace ae_accelsim::standalone
#endif
