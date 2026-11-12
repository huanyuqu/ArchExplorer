// SPDX-License-Identifier: Apache-2.0
#ifndef ARCH_EXPLORER_ACCELSIM_PARTITION_RANDOM_HPP
#define ARCH_EXPLORER_ACCELSIM_PARTITION_RANDOM_HPP

#include <array>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace ae_accelsim { namespace standalone {

// Reentrant form of the original libc rand/srand stream (128-byte state).
// Copying a configuration copies its state, never pointers into another one.
class PartitionRandom {
 public:
  PartitionRandom() { seed(1); }
  PartitionRandom(const PartitionRandom& other) { *this=other; }
  PartitionRandom& operator=(const PartitionRandom& other) {
    if(this==&other)return *this;
    storage_=other.storage_;data_=other.data_;
    data_.fptr=storage_.data()+(other.data_.fptr-other.storage_.data());
    data_.rptr=storage_.data()+(other.data_.rptr-other.storage_.data());
    data_.state=storage_.data()+(other.data_.state-other.storage_.data());
    data_.end_ptr=storage_.data()+(other.data_.end_ptr-other.storage_.data());
    return *this;
  }
  void seed(unsigned value) {
    data_={};storage_.fill(0);
    if(::initstate_r(value,reinterpret_cast<char*>(storage_.data()),sizeof(storage_),&data_))
      throw std::runtime_error("cannot initialize native partition RNG");
  }
  unsigned next() {
    std::int32_t value;
    if(::random_r(&data_,&value))throw std::runtime_error("native partition RNG failed");
    return static_cast<unsigned>(value);
  }
 private:
  std::array<std::int32_t,32> storage_{};
  struct random_data data_{};
};

}}  // namespace ae_accelsim::standalone
#endif
