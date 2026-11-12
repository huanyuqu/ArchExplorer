// Concrete Accel-Sim resource; owned by the component provider.
// SPDX-License-Identifier: Apache-2.0
// A file/stdin-only client of production's catalog parser and state rules.
// Never load the NVBit library or link a CUDA/driver/runtime library here.
#include "catalog_policy.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#ifndef AE_CATALOG_POLICY_SHA256
#error "Define AE_CATALOG_POLICY_SHA256 to the exact catalog_policy.hpp source SHA256"
#endif
#ifndef AE_CATALOG_HARNESS_SHA256
#error "Define AE_CATALOG_HARNESS_SHA256 to the exact catalog_harness.cpp source SHA256"
#endif

namespace {

template <std::size_t N>
constexpr bool source_sha256(const char (&value)[N]) {
    if (N != 65) return false;
    for (std::size_t i = 0; i < 64; ++i) {
        if (!((value[i] >= '0' && value[i] <= '9') ||
              (value[i] >= 'a' && value[i] <= 'f'))) return false;
    }
    return value[64] == '\0';
}

static_assert(source_sha256(AE_CATALOG_POLICY_SHA256), "policy SHA256 must be 64 lowercase hex bytes");
static_assert(source_sha256(AE_CATALOG_HARNESS_SHA256), "harness SHA256 must be 64 lowercase hex bytes");

}  // namespace

// Exact fixed-size allocatable, read-only ELF metadata, independent of debug
// symbols and compiler string merging. The caller inspects these sections offline.
extern "C" {
__attribute__((used, section(".ae_policy"), aligned(1)))
extern const char ae_catalog_policy_identity[65] = AE_CATALOG_POLICY_SHA256;
__attribute__((used, section(".ae_harness"), aligned(1)))
extern const char ae_catalog_harness_identity[65] = AE_CATALOG_HARNESS_SHA256;
}

namespace {

std::string quote(const std::string& value) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            result.push_back('\\');
            result.push_back(static_cast<char>(c));
        } else if (c < 32 || c >= 127) {
            result += "\\u00";
            result.push_back(hex[c >> 4]);
            result.push_back(hex[c & 15]);
        } else {
            result.push_back(static_cast<char>(c));
        }
    }
    result.push_back('"');
    return result;
}

ae_catalog::Catalog read_catalog(const std::filesystem::path& file) {
    if (!std::filesystem::is_regular_file(file)) {
        throw std::invalid_argument("catalog input must be a regular file");
    }
    std::ifstream input(file, std::ios::binary | std::ios::ate);
    if (!input) throw std::invalid_argument("catalog file could not be opened");
    const auto length = input.tellg();
    if (length <= 0 || length > static_cast<std::streamoff>(ae_catalog::detail::kMaximumBytes)) {
        throw std::invalid_argument("catalog file byte bound exceeded");
    }
    std::vector<unsigned char> bytes(static_cast<std::size_t>(length));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(bytes.data()), length) ||
        input.peek() != std::char_traits<char>::eof()) {
        throw std::invalid_argument("catalog file length changed or read failed");
    }
    return ae_catalog::Catalog::parse(bytes.data(), bytes.size());
}

void inspect(const ae_catalog::Catalog& catalog) {
    std::cout << "{\"schema\":\"arch-explorer.catalog-inspection.v1\",\"version\":"
              << catalog.version << ",\"count\":" << catalog.count
              << ",\"max_attempts\":" << catalog.max_attempts
              << ",\"record_capacity\":" << catalog.record_capacity
              << ",\"work_item\":" << quote(catalog.work_item)
              << ",\"registry_sha256\":" << quote(catalog.registry_sha256)
              << ",\"cubin_sha256\":" << quote(catalog.cubin_sha256)
              << ",\"policy_sha256\":" << quote(catalog.policy_sha256)
              << ",\"functions\":[";
    for (std::size_t i = 0; i < catalog.functions.size(); ++i) {
        std::cout << (i ? "," : "") << quote(catalog.functions[i]);
    }
    std::cout << "],\"catalog_sha256\":" << quote(catalog.sha256()) << "}\n";
}

// Reading commands is bounded as well as reading the catalog. On an overlong
// line the partial command is recorded and processing stops immediately.
bool line(std::istream& input, std::string& value) {
    value.clear();
    char c = 0;
    while (input.get(c)) {
        if (c == '\n') return true;
        if (value.size() >= 4096) throw std::invalid_argument("sequence command byte bound exceeded");
        value.push_back(c);
    }
    if (input.bad()) throw std::invalid_argument("sequence stdin read failed");
    return !value.empty();
}

void command(ae_catalog::Gate& gate, const std::string& text) {
    std::istringstream fields(text);
    std::string verb, argument, extra;
    fields >> verb;
    if (verb == "begin") {
        if (!(fields >> argument) || fields >> extra) {
            throw std::invalid_argument("begin requires exactly one function name");
        }
        gate.begin(argument);
    } else if (verb == "launch" || verb == "finish") {
        if (fields >> extra) throw std::invalid_argument("launch/finish take no arguments");
        if (verb == "launch") gate.launch();
        else gate.finish();
    } else {
        throw std::invalid_argument("unknown sequence command; expected begin/launch/finish");
    }
}

void state(const ae_catalog::Gate& gate, const std::string& text, const char* status,
           const std::string& reason = {}) {
    std::cout << "{\"command\":" << quote(text) << ",\"status\":" << quote(status)
              << ",\"attempts\":" << gate.attempts()
              << ",\"launched_total\":" << gate.launched_total()
              << ",\"armed\":" << (gate.armed() ? "true" : "false");
    if (!reason.empty()) std::cout << ",\"reason\":" << quote(reason);
    std::cout << "}\n";
}

int sequence(ae_catalog::Catalog catalog) {
    ae_catalog::Gate gate(std::move(catalog));
    for (;;) {
        std::string text;
        try {
            if (!line(std::cin, text)) return 0;
            command(gate, text);
            state(gate, text, "passed");
        } catch (const std::exception& error) {
            state(gate, text, "error", error.what());
            return 2;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "inspect") {
            inspect(read_catalog(argv[2]));
        } else if (argc == 4 && std::string(argv[1]) == "check") {
            const auto expected = read_catalog(argv[2]);
            const auto actual = read_catalog(argv[3]);
            ae_catalog::require_compatible(expected, actual);
            std::cout << "{\"schema\":\"arch-explorer.catalog-compatibility.v1\","
                      << "\"status\":\"passed\",\"catalog_sha256\":"
                      << quote(expected.sha256()) << "}\n";
        } else if (argc == 3 && std::string(argv[1]) == "sequence") {
            return sequence(read_catalog(argv[2]));
        } else {
            throw std::invalid_argument("usage: catalog_harness inspect BIN | check EXPECTED ACTUAL | sequence BIN");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cout << "{\"schema\":\"arch-explorer.catalog-harness-error.v1\","
                  << "\"status\":\"failed\",\"reason\":" << quote(error.what()) << "}\n";
        return 2;
    }
}
