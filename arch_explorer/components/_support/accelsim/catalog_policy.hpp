// Concrete Accel-Sim resource; owned by the component provider.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// This header owns the finite capture directory format and begin/launch/finish
// rules. It reads bytes only and has no CUDA, filesystem, process or lock API.
// Catalog names are parsed from the hash-bound byte contract; there is
// no second, separately maintained function-name array.
#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ae_catalog {
namespace detail {

inline constexpr std::size_t kMaximumBytes = 2U * 1024U * 1024U;
inline constexpr std::uint32_t kMaximumCount = 4096;
inline constexpr std::array<unsigned char, 16> kMagic = {
    'A', 'R', 'C', 'H', 'C', 'A', 'T', 'A', 'L', 'O', 'G', '0', '0', '0', '1', 0};

inline void require(bool condition, const char* reason) {
    if (!condition) throw std::invalid_argument(reason);
}

inline std::array<unsigned char, SHA256_DIGEST_LENGTH> digest(
        const unsigned char* bytes, std::size_t size) {
    std::array<unsigned char, SHA256_DIGEST_LENGTH> result{};
    require(SHA256(bytes, size, result.data()) != nullptr,
            "catalog SHA256 calculation failed");
    return result;
}

inline std::string hex(const std::array<unsigned char, SHA256_DIGEST_LENGTH>& digest) {
    static constexpr char alphabet[] = "0123456789abcdef";
    std::string result;
    result.reserve(2 * digest.size());
    for (unsigned char byte : digest) {
        result.push_back(alphabet[byte >> 4]);
        result.push_back(alphabet[byte & 15]);
    }
    return result;
}

inline bool identifier_initial(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}

inline bool identifier(const std::string& name) {
    if (name.empty() || !identifier_initial(static_cast<unsigned char>(name[0]))) return false;
    return std::all_of(name.begin() + 1, name.end(), [](unsigned char c) {
        return identifier_initial(c) || (c >= '0' && c <= '9');
    });
}

class Cursor {
 public:
    Cursor(const unsigned char* bytes, std::size_t size) : bytes_(bytes), size_(size) {}

    const unsigned char* take(std::size_t size) {
        require(size <= size_ - position_, "catalog is truncated");
        const auto* result = bytes_ + position_;
        position_ += size;
        return result;
    }

    std::uint16_t u16() {
        const auto* p = take(2);
        return static_cast<std::uint16_t>(p[0]) |
               static_cast<std::uint16_t>(static_cast<std::uint16_t>(p[1]) << 8);
    }

    std::uint32_t u32() {
        const auto* p = take(4);
        return static_cast<std::uint32_t>(p[0]) |
               (static_cast<std::uint32_t>(p[1]) << 8) |
               (static_cast<std::uint32_t>(p[2]) << 16) |
               (static_cast<std::uint32_t>(p[3]) << 24);
    }

    std::string text(std::size_t size) {
        return std::string(reinterpret_cast<const char*>(take(size)), size);
    }

    std::string sha256_text() {
        auto value = text(64);
        require(std::all_of(value.begin(), value.end(), [](unsigned char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        }), "catalog identity must be exactly 64 lowercase hexadecimal bytes");
        return value;
    }

    bool exhausted() const noexcept { return position_ == size_; }

 private:
    const unsigned char* bytes_;
    std::size_t size_;
    std::size_t position_ = 0;
};

}  // namespace detail

struct Catalog {
    std::uint32_t version = 0, count = 0, max_attempts = 0, record_capacity = 0;
    std::string registry_sha256, cubin_sha256, policy_sha256, work_item;
    std::vector<std::string> functions;
    std::vector<unsigned char> bytes;

    static Catalog parse(const unsigned char* input, std::size_t size) {
        // Fixed header, three SHA strings, work-item length and trailing digest.
        // Variable work item and at least one name are checked by the cursor.
        detail::require(input != nullptr, "catalog input is null");
        detail::require(size >= 16 + 5 * 4 + 3 * 64 + 2 + 32 &&
                        size <= detail::kMaximumBytes, "catalog byte bound exceeded");
        detail::Cursor cursor(input, size - SHA256_DIGEST_LENGTH);
        detail::require(std::memcmp(cursor.take(detail::kMagic.size()),
                                    detail::kMagic.data(), detail::kMagic.size()) == 0,
                        "unknown catalog magic");
        Catalog result;
        result.version = cursor.u32();
        detail::require(result.version == 1, "unknown catalog version");
        detail::require(cursor.u32() == size, "catalog total_bytes differs from exact input");
        result.count = cursor.u32();
        result.max_attempts = cursor.u32();
        result.record_capacity = cursor.u32();
        detail::require(result.count >= 1 && result.count <= detail::kMaximumCount,
                        "catalog function count must be in 1..4096");
        detail::require(result.max_attempts == result.count,
                        "catalog max_attempts must equal function count");
        detail::require(result.record_capacity >= 1 && result.record_capacity <= 4096,
                        "catalog record_capacity must be in 1..4096");
        result.registry_sha256 = cursor.sha256_text();
        result.cubin_sha256 = cursor.sha256_text();
        result.policy_sha256 = cursor.sha256_text();
        const auto work_size = cursor.u16();
        detail::require(work_size >= 1 && work_size <= 128,
                        "catalog work_item length must be in 1..128");
        result.work_item = cursor.text(work_size);
        detail::require(std::all_of(result.work_item.begin(), result.work_item.end(),
                        [](unsigned char c) { return c >= 32 && c <= 126; }),
                        "catalog work_item must be printable ASCII");
        result.functions.reserve(result.count);
        std::set<std::string> unique;
        for (std::uint32_t i = 0; i < result.count; ++i) {
            const auto name_size = cursor.u16();
            detail::require(name_size >= 1 && name_size <= 255,
                            "catalog function-name length must be in 1..255");
            auto name = cursor.text(name_size);
            detail::require(detail::identifier(name),
                            "catalog function name must be an ASCII C identifier");
            detail::require(unique.insert(name).second, "duplicate catalog function name");
            result.functions.push_back(std::move(name));
        }
        detail::require(cursor.exhausted(), "catalog has padding or undeclared fields");
        const auto payload_digest = detail::digest(input, size - SHA256_DIGEST_LENGTH);
        detail::require(std::memcmp(payload_digest.data(), input + size - SHA256_DIGEST_LENGTH,
                                    SHA256_DIGEST_LENGTH) == 0,
                        "catalog trailing SHA256 differs from payload bytes");
        result.bytes.assign(input, input + size);
        result.parsed_sha256_ = detail::hex(detail::digest(input, size));
        return result;
    }

    std::string sha256() const {
        require_parsed();
        return parsed_sha256_;
    }

 private:
    // Catalog's required public fields are convenient views, never independent
    // authority. A caller cannot default-construct or mutate a view/byte buffer
    // and have it treated as a parsed contract by compatibility or Gate.
    std::string parsed_sha256_;

    void require_parsed() const {
        detail::require(!parsed_sha256_.empty(), "catalog must first be parsed");
        const auto canonical = parse(bytes.data(), bytes.size());
        detail::require(canonical.parsed_sha256_ == parsed_sha256_ &&
                        version == canonical.version && count == canonical.count &&
                        max_attempts == canonical.max_attempts &&
                        record_capacity == canonical.record_capacity &&
                        registry_sha256 == canonical.registry_sha256 &&
                        cubin_sha256 == canonical.cubin_sha256 &&
                        policy_sha256 == canonical.policy_sha256 &&
                        work_item == canonical.work_item && functions == canonical.functions,
                        "catalog bytes or parsed fields changed after parsing");
    }

    friend void require_compatible(const Catalog&, const Catalog&);
    friend class Gate;
};

inline void require_compatible(const Catalog& expected, const Catalog& actual) {
    expected.require_parsed();
    actual.require_parsed();
    detail::require(expected.bytes == actual.bytes,
                    "catalog differs from complete expected byte binding");
}

class Gate {
 public:
    explicit Gate(Catalog catalog) : catalog_(std::move(catalog)) {
        catalog_.require_parsed();
    }

    void check_begin(const std::string& function) const {
        detail::require(!armed_, "begin requires idle capture state");
        detail::require(attempted_.size() < catalog_.max_attempts,
                        "catalog attempt budget exhausted");
        detail::require(std::find(catalog_.functions.begin(), catalog_.functions.end(), function) !=
                        catalog_.functions.end(), "function is absent from complete catalog");
        detail::require(attempted_.count(function) == 0,
                        "function was already attempted; retry is forbidden");
    }

    void begin(const std::string& function) {
        check_begin(function);
        attempted_.insert(function);
        armed_ = true;
        launched_ = false;
    }

    void launch() {
        detail::require(armed_ && !launched_,
                        "launch requires an armed, not-yet-launched attempt");
        launched_ = true;
        ++launched_total_;
    }

    void finish() {
        detail::require(armed_ && launched_, "finish requires the armed attempt to launch");
        armed_ = false;
        launched_ = false;
        // attempted_ deliberately survives finish: there are no replacements.
    }

    std::size_t attempts() const noexcept { return attempted_.size(); }
    std::size_t launched_total() const noexcept { return launched_total_; }
    bool armed() const noexcept { return armed_; }

 private:
    Catalog catalog_;
    std::set<std::string> attempted_;
    std::size_t launched_total_ = 0;
    bool armed_ = false;
    bool launched_ = false;
};

}  // namespace ae_catalog
