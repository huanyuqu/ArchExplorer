"""Extract instance-owned BookSim/LocalInterconnect, without a GPU host.

Run BEFORE SIM38's peek/extract, on the fresh typed-v1 source. Originals are
read only. Output is a separate library under intersim2/ae_standalone. Every
mutable native dependency is passed explicitly; no current-instance globals,
thread-local dispatch or per-network copies of a GPU context are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

TREE_SHA256 = '959981c7de85427a29b92aac7205d2a14206a33898924ed9d83fdcc82da89e57'
LOCAL_SHA256 = {
    'local_interconnect.h': 'a67a6b02c1e236d3bc36100ac729d45f4e2bdaeec0db10a7a42244505d449d1a',
    'local_interconnect.cc': '02146e90be68286ba1620fda8f4079c59cae2df5160fc840b3181e5075a3cbaf',
}
NS = 'ae_standalone_booksim'
GLOBALS = ('gPrintActivity', 'gK', 'gN', 'gC', 'gNodes', 'gTrace', 'gWatchOut',
           'gNumVCs', 'gReadReqBeginVC', 'gReadReqEndVC', 'gWriteReqBeginVC',
           'gWriteReqEndVC', 'gReadReplyBeginVC', 'gReadReplyEndVC',
           'gWriteReplyBeginVC', 'gWriteReplyEndVC', 'gRoutingFunctionMap')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'expected one ICNT extraction anchor: {old[:100]!r}')
    return text.replace(old, new, 1)


def cpp_code(text):
    """Mask comments/literals without moving C++ token offsets or lines."""
    return re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                  lambda m: re.sub(r'[^\n]', ' ', m[0]), text, flags=re.S)


def constructor_declaration(text, klass, parameters):
    """Change the unique empty constructor declaration, never its destructor.

    Bare ``Class();`` is also a substring of ``~Class();``. Anchor the entire
    declaration in tokenized code and replace only the parameter span. Pinned
    declarations are standalone lines; qualified definitions are handled
    separately with their complete class-qualified signature.
    """
    mask = cpp_code(text)
    matches = list(re.finditer(r'(?m)^[ \t]*(?:explicit[ \t]+)?' + re.escape(klass) +
                               r'[ \t]*\((?P<parameters>\s*)\)[ \t]*;', mask))
    if len(matches) != 1:
        raise ValueError(f'expected one complete empty constructor declaration: {klass}')
    start, end = matches[0].span('parameters')
    return text[:start] + parameters + text[end:]


def special_member_signatures(header, implementation, klass):
    """Require a matched constructor/destructor pair for the state-owned class."""
    h, c = cpp_code(header), cpp_code(implementation)
    name = re.escape(klass)
    patterns = {
        'constructor_declaration': (h, r'(?m)^[ \t]*(?:explicit[ \t]+)?' + name + r'\s*\(([^()]*)\)\s*;'),
        'destructor_declaration': (h, r'(?m)^[ \t]*(?:virtual[ \t]+)?~\s*' + name + r'\s*\(([^()]*)\)\s*;'),
        'constructor_definition': (c, r'(?<![\w:~])' + name + r'::' + name + r'\s*\(([^()]*)\)'),
        'destructor_definition': (c, r'(?<![\w:~])' + name + r'::~\s*' + name + r'\s*\(([^()]*)\)'),
    }
    signatures = {}
    for kind, (code, pattern) in patterns.items():
        matches = list(re.finditer(pattern, code))
        if len(matches) != 1:
            raise ValueError(f'expected one {klass} {kind}')
        signatures[kind] = re.sub(r'\s+', '', matches[0][1])
    for kind in ('constructor', 'destructor'):
        if signatures[kind + '_declaration'] != signatures[kind + '_definition']:
            raise ValueError(f'{klass} {kind} declaration/definition mismatch')
    if signatures['constructor_declaration'] != 'AeBookSimState&state':
        raise ValueError(f'{klass} constructor must explicitly receive its owned state')
    if signatures['destructor_declaration']:
        raise ValueError(f'{klass} destructor must have no parameters')
    return signatures


def audit_destructors(code):
    """Check all emitted classes, including unchanged support/allocator types."""
    classes = set()
    for text in code.values():
        classes.update(re.findall(r'\b(?:class|struct)\s+(\w+)', cpp_code(text)))
    signatures = []
    for name, text in sorted(code.items()):
        masked = cpp_code(text)
        for match in re.finditer(r'(?<!\w)~\s*(\w+)\s*\(([^()]*)\)', masked):
            if match[1] not in classes:
                continue
            parameters = re.sub(r'\s+', '', match[2])
            if parameters not in ('', 'void'):
                raise ValueError(f'parameterized destructor in {name}: {match[1]}')
            signatures.append({'file': name, 'class': match[1], 'parameters': parameters})
    return signatures


def local_boundary_member(header, implementation):
    """Use the real LocalInterconnect field, not the xbar/BookSim spelling."""
    _, _, declaration = block(header, 'class LocalInterconnect {')
    fields = set()
    for match in re.finditer(r'(?m)^[ \t]*unsigned[ \t]+([^;(){}]+);', cpp_code(declaration)):
        for field in match[1].split(','):
            name = re.fullmatch(r'\s*(\w+)\s*', field)
            if name:
                fields.add(name[1])
    boundary = None
    for method, argument, comparison in [('HasPacket', 'deviceID', '<'),
                                          ('HasBuffer', 'deviceID', '>='),
                                          ('Pop', 'ouput_deviceID', '<')]:
        _, _, native = block(implementation, 'LocalInterconnect::' + method + '(')
        uses = re.findall(r'\b' + argument + r'\s*' + comparison + r'\s*(\w+)', cpp_code(native))
        if len(uses) != 1 or uses[0] not in fields:
            raise ValueError(f'LocalInterconnect::{method} must use a declared endpoint boundary')
        if boundary is not None and boundary != uses[0]:
            raise ValueError('LocalInterconnect endpoint boundaries disagree')
        boundary = uses[0]
    return boundary


STANDARD_USES = {'cassert': r'\bassert\s*\(', 'set': r'\b(?:std::)?set\s*<'}


def direct_standard_declarations(text):
    """Own the standard declarations previously supplied by a removed GPU TU."""
    masked = cpp_code(text)
    required = [header for header, pattern in STANDARD_USES.items() if re.search(pattern, masked)]
    missing = [header for header in required
               if not re.search(r'^\s*#\s*include\s*<' + header + '>', masked, re.M)]
    return ''.join('#include <' + header + '>\n' for header in missing) + text, required


def block(text, marker):
    """Find a unique C++ brace block, ignoring comments/strings."""
    if text.count(marker) != 1:
        raise ValueError(f'non-unique block: {marker}')
    anchor = text.index(marker)
    # Class-definition anchors include their opening brace to distinguish
    # them from native forward/friend declarations. Function anchors do not.
    if marker.rstrip().endswith('{'):
        pos = anchor + len(marker.rstrip()) - 1
    else:
        pos = text.index('{', anchor + len(marker))
    depth = 0
    for m in re.finditer(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]', text[pos:], re.S):
        if m[0] == '{':
            depth += 1
        elif m[0] == '}':
            depth -= 1
            if not depth:
                return pos, pos + m.end(), text[pos + 1:pos + m.start()]
    raise ValueError(f'unterminated block: {marker}')


def body(text, marker, replacement):
    a, b, _ = block(text, marker)
    return text[:a] + '{\n' + replacement + '\n}' + text[b:]


def function(text, marker):
    a, b, _ = block(text, marker)
    return text[text.index(marker):b] + '\n'


def bind_globals(text, owner='this->AeState()'):
    for name in sorted(GLOBALS, key=len, reverse=True):
        text = re.sub(r'\b' + name + r'\b', owner + '.' + name, text)
    return text


def close_base_declarations(code):
    """Make base-class dependencies direct after narrowing routefunc.hpp.

    The pinned native VC/Buffer headers obtained Module transitively through
    the old routing header. The supported routing interface no longer includes
    every router topology. Preserve that narrow interface and include actual
    base declarations where they are used, across the full selected header set.
    """
    headers={n:t for n,t in code.items() if Path(n).suffix in ('.hpp','.h')}
    definitions={}
    for name,text in headers.items():
        for match in re.finditer(r'\b(?:class|struct)\s+(\w+)\s*(?::[^;{]+)?\{',text):
            definitions[match[1]]=name
    for name,text in headers.items():
        dependencies=set()
        for match in re.finditer(r'\b(?:class|struct)\s+\w+\s*:\s*([^;{]+)\{',text):
            for base in match[1].split(','):
                base=re.sub(r'\b(?:public|protected|private|virtual)\b','',base).strip()
                base=re.match(r'\w+',base)
                if base and base[0] in definitions:
                    target=definitions[base[0]]
                    if target!=name and not re.search(r'#\s*include\s+"'+re.escape(Path(target).name)+'"',text):
                        dependencies.add(target)
        if dependencies:
            code[name]=''.join('#include "'+Path(dep).name+'"\n' for dep in sorted(dependencies))+text
    return code


def namespace(text):
    # BookSim uses conventional file-level includes only. Keep ALL includes
    # outside the namespace (also template headers); rename guards to prevent
    # accidental suppression when linked beside the untouched oracle library.
    includes = re.findall(r'^\s*#\s*include[^\n]+', text, re.M)
    text = re.sub(r'^\s*#\s*include[^\n]+', '', text, flags=re.M)
    text = re.sub(r'\b(_[A-Z][A-Z0-9_]*_(?:HPP|H)_)\b', r'AE_STANDALONE\1', text)
    return '\n'.join(includes) + f'\nnamespace {NS} {{\n' + text + f'\n}} // {NS}\n'


def native_state(raw):
    rng = []
    for name, suffix, method in [('rng.c', 'I', 'long next_int() { return ran_arr_next(); }'),
                                  ('rng-double.c', 'D', 'double next_float() { return ranf_arr_next(); }')]:
        code = raw[name].split('#include <stdio.h>')[0]
        # Preserve Knuth's recurrence and seed/warm-up byte-for-byte. Only the
        # state declarations move inside this instance; no replacement RNG.
        code = re.sub(r'^(long ran_x\[KK\]|long ran_arr_buf\[QUALITY\]|double ran_u\[KK\]|double ranf_arr_buf\[QUALITY\]);', r'\1 = {};', code, flags=re.M)
        macros = re.findall(r'^#define\s+(\w+)', code, re.M)
        rng.append(code + '\n' + method + '\n' + '\n'.join('#undef ' + m for m in macros))
    return '''#ifndef AE_BOOKSIM_STATE_HPP
#define AE_BOOKSIM_STATE_HPP
#include <map>
#include <stack>
#include <string>
#include <ostream>
#include <stdexcept>
class Router; class Flit; class Credit; class PacketReplyInfo; class OutputSet;
struct AeBookSimState;
typedef void (*tRoutingFunction)(AeBookSimState&, const Router*, const Flit*, int, OutputSet*, bool);
struct AeBookSimState {
  AeBookSimState() = default;
  ~AeBookSimState();
  AeBookSimState(const AeBookSimState&) = delete;
  AeBookSimState& operator=(const AeBookSimState&) = delete;
  int* clock = nullptr;
  bool gPrintActivity = false, gTrace = false;
  std::ostream* gWatchOut = nullptr;
  int gK = 0, gN = 0, gC = 0, gNodes = 0, gNumVCs = 0;
  int gReadReqBeginVC = 0, gReadReqEndVC = 0;
  int gWriteReqBeginVC = 0, gWriteReqEndVC = 0;
  int gReadReplyBeginVC = 0, gReadReplyEndVC = 0;
  int gWriteReplyBeginVC = 0, gWriteReplyEndVC = 0;
  std::map<std::string,tRoutingFunction> gRoutingFunctionMap;
  std::stack<Flit*> flit_all, flit_free;
  std::stack<Credit*> credit_all, credit_free;
  std::stack<PacketReplyInfo*> reply_all, reply_free;
''' + '\n'.join(rng) + '\n};\n#endif\n'


PARSER = r'''
void Configuration::ParseFile(string const& filename) {
  std::ifstream in(filename.c_str(), std::ios::binary);
  if(!in) throw std::invalid_argument("cannot read BookSim config: " + filename);
  std::ostringstream out; out << in.rdbuf();
  if(in.bad() || out.str().size() > 1024*1024)
    throw std::invalid_argument("BookSim config exceeds 1MiB/read failed");
  ParseString(out.str());
}
void Configuration::ParseString(string const& input) {
  // Reentrant implementation of config.y's three assignment productions.
  // A token is an integer, finite float, or a string (including nested arrays).
  std::string text; text.reserve(input.size());
  for(size_t i=0;i<input.size();) {
    if(input.compare(i,2,"//")==0) {
      while(i<input.size() && input[i]!='\n') ++i;
    } else if(input.compare(i,2,"/*")==0) {
      auto end=input.find("*/",i+2);
      if(end==string::npos) ParseError("unterminated comment");
      i=end+2;
    } else text += input[i++];
  }
  size_t pos=0;
  auto skip=[&]() { while(pos<text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos; };
  while(true) {
    skip(); if(pos==text.size()) break;
    size_t start=pos;
    while(pos<text.size() && (std::isalnum(static_cast<unsigned char>(text[pos])) || text[pos]=='_')) ++pos;
    if(pos==start) ParseError("expected configuration field");
    string field=text.substr(start,pos-start); skip();
    if(pos==text.size() || text[pos++]!='=') ParseError("expected =");
    skip(); start=pos; int braces=0;
    while(pos<text.size() && (text[pos]!=';' || braces)) {
      if(text[pos]=='{') ++braces;
      if(text[pos]=='}' && --braces<0) ParseError("unbalanced array");
      ++pos;
    }
    if(braces || pos==text.size()) ParseError("expected terminating ;");
    size_t end=pos++;
    while(end>start && std::isspace(static_cast<unsigned char>(text[end-1]))) --end;
    string value=text.substr(start,end-start);
    if(value.empty()) ParseError("empty configuration value");
    // Match config.l: array and bare identifier values use Assign(string).
    const bool numeric = std::isdigit(static_cast<unsigned char>(value[0])) || value[0]=='-' || value[0]=='.';
    if(!numeric) { Assign(field,value); continue; }
    try {
      size_t used=0;
      if(value.find_first_of(".eE")!=string::npos) {
        double number=std::stod(value,&used);
        if(used!=value.size() || !std::isfinite(number)) ParseError("invalid finite float");
        Assign(field,number);
      } else {
        long long number=std::stoll(value,&used,10);
        if(used!=value.size() || number<INT_MIN || number>INT_MAX) ParseError("invalid integer");
        Assign(field,static_cast<int>(number));
      }
    } catch(const std::exception&) { ParseError("invalid numeric value for " + field); }
  }
}
'''


VALIDATION_POLICY = {
    'fixed': {
        'int': {
            'subnets': 2, 'classes': 1, 'use_map': 0,
            'read_request_subnet': 0, 'write_request_subnet': 0,
            'read_reply_subnet': 1, 'write_reply_subnet': 1,
            'link_failures': 0, 'sim_power': 0, 'noq': 0, 'speculative': 0,
            'perfect_icnt': 0, 'fixed_lat_per_hop': 0,
        },
        'str': {
            'topology': 'fly', 'routing_function': 'dest_tag', 'router': 'iq',
            'sim_type': 'gpgpusim', 'vc_allocator': 'islip', 'sw_allocator': 'islip',
            'traffic': 'uniform', 'injection_process': 'bernoulli',
            'watch_file': '', 'watch_out': '', 'stats_out': '',
        },
    },
    'choices': {'str': {
        'packet_size': ['', '{{1,2,3,4},{10,20}}'],
        'packet_size_rate': ['', '{{1,1,1,1},{2,1}}'],
    }},
    'variable': {
        'int': ['k', 'n', 'flit_size', 'network_count', 'num_vcs', 'vc_buf_size',
                'input_buffer_size', 'ejection_buffer_size', 'boundary_buffer_size',
                'wait_for_tail_credit', 'alloc_iters', 'credit_delay', 'routing_delay',
                'vc_alloc_delay', 'sw_alloc_delay', 'input_speedup', 'output_speedup',
                'st_prepare_delay', 'st_final_delay', 'seed',
                'read_request_begin_vc', 'read_request_end_vc',
                'write_request_begin_vc', 'write_request_end_vc',
                'read_reply_begin_vc', 'read_reply_end_vc',
                'write_reply_begin_vc', 'write_reply_end_vc'],
        'float': ['internal_speedup'],
    },
}


def validation_policy_cpp():
    """One field table owns fixed constraints AND their default exemptions."""
    policy = VALIDATION_POLICY
    allowed = {}
    for kind in ('int', 'float', 'str'):
        fixed = set(policy['fixed'].get(kind, {}))
        choices = set(policy['choices'].get(kind, {}))
        variable = set(policy['variable'].get(kind, []))
        if fixed & choices or fixed & variable or choices & variable:
            raise ValueError('overlapping ICNT validation policy: ' + kind)
        allowed[kind] = fixed | choices | variable
    def literal(value):
        return json.dumps(value, ensure_ascii=True)
    def keys(values):
        return '{' + ','.join(literal(key) for key in sorted(values)) + '}'
    lines = [
        '  const std::set<std::string> ints=' + keys(allowed['int']) + ';',
        '  const std::set<std::string> floats=' + keys(allowed['float']) + ';',
        '  const std::set<std::string> strings=' + keys(allowed['str']) + ';',
        '  for(const auto& item : c.GetIntMap()) require(ints.count(item.first) || item.second==defaults.GetInt(item.first),std::string("nondefault unextracted integer policy: ")+item.first);',
        '  for(const auto& item : c.GetFloatMap()) require(floats.count(item.first) || item.second==defaults.GetFloat(item.first),std::string("nondefault unextracted floating policy: ")+item.first);',
        '  for(const auto& item : c.GetStrMap()) require(strings.count(item.first) || item.second==defaults.GetStr(item.first),std::string("nondefault unextracted string policy: ")+item.first);',
    ]
    for kind, getter in (('int', 'GetInt'), ('str', 'GetStr')):
        for key, value in sorted(policy['fixed'].get(kind, {}).items()):
            expected = literal(value)
            message = literal(key + ': expected ' + str(value))
            lines.append(f'  require(c.{getter}({literal(key)})=={expected},{message});')
    for key, values in sorted(policy['choices']['str'].items()):
        tests = ' || '.join('c.GetStr(' + literal(key) + ')==' + literal(value) for value in values)
        lines.append('  require(' + tests + ',' + literal(key + ': unsupported declaration') + ');')
    return '\n'.join(lines) + '\n'


def audit_validation_policy(implementation):
    _, _, native = block(implementation, 'void InterconnectInterface::AeValidate(')
    expected = validation_policy_cpp()
    if native.count(expected) != 1:
        raise ValueError('ICNT fixed constraints and default exemptions must share their field table')
    return sha(expected.encode())


VALIDATE = r'''
void InterconnectInterface::AeValidate(unsigned requests, unsigned memories) const {
  const Configuration& c = *_icnt_config;
  auto require=[](bool value,const std::string& msg) { if(!value) throw std::invalid_argument(msg); };
  // Reject unextracted modes before ANY router/traffic-manager allocation.
  // Other native policy defaults remain authoritative; only the following
  // documented dimensions/timing/buffer/VC choices may differ from defaults.
  IntersimConfig defaults;
  // @field_policy
  require(requests && memories && requests+uint64_t(memories)<=4096, "request_nodes/memory_nodes: ICNT endpoint bound");
  require(c.GetInt("k")>=2 && c.GetInt("k")<=4096, "k: fly dimension out of bounds");
  require(c.GetInt("n")>=1 && c.GetInt("n")<=8, "n: fly dimension out of bounds");
  uint64_t nodes=1; for(int i=0;i<c.GetInt("n");++i) { nodes*=c.GetInt("k"); require(nodes<=4096,"k/n: fly node bound"); }
  require(nodes>=requests+memories,"k/n/request_nodes/memory_nodes: fly has fewer nodes than endpoints");
  require(uint64_t(c.GetInt("n"))*nodes*c.GetInt("k")*2<=1048576,"k/n: fly crossbar resource admission bound");
  require(c.GetInt("flit_size")>0 && c.GetInt("flit_size")<=1048576, "flit_size: out of bounds");
  require(c.GetInt("num_vcs")>0 && c.GetInt("num_vcs")<=64, "num_vcs: out of bounds");
  require(c.GetInt("vc_buf_size")>0 && c.GetInt("vc_buf_size")<=1048576, "vc_buf_size: out of bounds");
  require(c.GetInt("input_buffer_size")>=0 && c.GetInt("input_buffer_size")<=1048576, "input_buffer_size: out of bounds");
  require(c.GetInt("ejection_buffer_size")>=0 && c.GetInt("ejection_buffer_size")<=1048576, "ejection_buffer_size: out of bounds");
  require(c.GetInt("boundary_buffer_size")>0 && c.GetInt("boundary_buffer_size")<=1048576, "boundary_buffer_size: out of bounds");
  for(const char* field : {"routing_delay","credit_delay","st_prepare_delay","st_final_delay"}) require(c.GetInt(field)>=0 && c.GetInt(field)<=1024,std::string(field)+": negative/excessive ICNT delay");
  for(const char* field : {"vc_alloc_delay","sw_alloc_delay","input_speedup","output_speedup","alloc_iters"}) require(c.GetInt(field)>0 && c.GetInt(field)<=1024,std::string(field)+": invalid IQ allocation timing");
  require(std::isfinite(c.GetFloat("internal_speedup")) && c.GetFloat("internal_speedup")>0 && c.GetFloat("internal_speedup")<=1024, "internal_speedup: finite value required in (0,1024]");
  const int vcs=c.GetInt("num_vcs");
  for(auto range : {std::make_pair("read_request_begin_vc","read_request_end_vc"),std::make_pair("write_request_begin_vc","write_request_end_vc"),std::make_pair("read_reply_begin_vc","read_reply_end_vc"),std::make_pair("write_reply_begin_vc","write_reply_end_vc")}) {
    int begin=c.GetInt(range.first), end=c.GetInt(range.second);
    const bool reply=std::string(range.first).find("reply")!=std::string::npos;
    if(begin<0) begin=reply ? vcs/2 : 0;
    if(end<0) end=reply ? vcs-1 : vcs/2-1;
    require(begin>=0 && begin<=end && end<vcs,std::string(range.first)+"/"+range.second+": invalid native VC range");
  }
}
'''

VALIDATE = once(VALIDATE, '  // @field_policy\n', validation_policy_cpp())


def transform(raw):
    code = dict(raw)
    # All native configuration and Module descendants receive an explicit
    # network-owned state; ordinary parent-child native construction propagates it.
    code['booksim.hpp'] = '#include <stdexcept>\n#include <cstdint>\n#include "ae_state.hpp"\n' + code['booksim.hpp']
    h = code['module.hpp']
    h = once(h, 'class Module {', 'class Module {\n  AeBookSimState* _ae_state;')
    h = once(h, 'Module( Module *parent, const string& name );', '''Module( Module *parent, const string& name, AeBookSimState* state = nullptr );
  AeBookSimState& AeState() const { return *_ae_state; }
  int GetSimTime() const { return AeState().clock ? *AeState().clock : 0; }''')
    code['module.hpp'] = h
    code['module.cpp'] = once(code['module.cpp'], 'Module::Module( Module *parent, const string& name )',
                             'Module::Module( Module *parent, const string& name, AeBookSimState* state )\n  : _ae_state(parent ? &parent->AeState() : state)')
    code['module.cpp'] = once(code['module.cpp'], '  _name = name;', '  if(!_ae_state) throw std::logic_error("BookSim module without owner");\n  _name = name;')
    code['module.cpp'] = body(code['module.cpp'], 'void Module::Error(', '  throw std::runtime_error("BookSim " + _fullname + ": " + msg);')
    code['timed_module.hpp'] = once(code['timed_module.hpp'], 'TimedModule(Module * parent, string const & name) : Module(parent, name)',
                                   'TimedModule(Module * parent, string const & name, AeBookSimState* state = nullptr) : Module(parent, name, state)')
    ch = code['config_utils.hpp']
    ch = once(ch, 'int yyparse();', '')
    ch = once(ch, '  static Configuration * theConfig;', '  AeBookSimState* _ae_state = nullptr;')
    ch = once(ch, '  static Configuration * GetTheConfig();', '''  void AeBindState(AeBookSimState& state) { _ae_state=&state; }
  AeBookSimState& AeState() const { if(!_ae_state) throw std::logic_error("BookSim config without owner"); return *_ae_state; }''')
    code['config_utils.hpp'] = ch
    cc = once(code['config_utils.cpp'], 'Configuration *Configuration::theConfig = 0;', '')
    cc = once(cc, '  theConfig = this;', '')
    start = cc.index('void Configuration::ParseFile(')
    end = cc.index('bool ParseArgs(', start)
    cc = cc[:start] + PARSER + '\nvoid Configuration::ParseError(string const& msg, unsigned int line) const { throw std::invalid_argument("BookSim config: " + msg); }\n' + cc[end:]
    code['config_utils.cpp'] = '#include <cctype>\n#include <cmath>\n' + cc
    code['globals.hpp'] = '#include "ae_state.hpp"\n'  # no singleton declarations
    code['networks/network.cpp'] = code['networks/network.cpp'].replace('TimedModule( 0, name )', 'TimedModule( 0, name, &config.AeState() )')
    code['networks/network.cpp'] = body(code['networks/network.cpp'], 'Network * Network::New(', '''  if(config.GetStr("topology") != "fly") throw std::invalid_argument("unsupported native topology");
  return new KNFly(config,name);''')
    for name in ('kncube','cmesh','flatfly_onchip','qtree','tree4','fattree','anynet','dragonfly'):
        code['networks/network.cpp'] = code['networks/network.cpp'].replace(f'#include "{name}.hpp"', '')
    code['routers/router.cpp'] = code['routers/router.cpp'].replace('#include "event_router.hpp"','').replace('#include "chaos_router.hpp"','')
    code['routers/router.cpp'] = body(code['routers/router.cpp'], 'Router *Router::NewRouter(', '''  if(config.GetStr("router") != "iq") throw std::invalid_argument("unsupported native router");
  return new IQRouter(config,parent,name,id,inputs,outputs);''')
    tm = code['trafficmanager.cpp'].replace('#include "batchtrafficmanager.hpp"','')
    tm = body(tm, 'TrafficManager * TrafficManager::New(', '  return new GPUTrafficManager(config,net);')
    tm = once(tm, 'Module( 0, "traffic_manager" )', 'Module( 0, "traffic_manager", &config.AeState() )')
    tm = once(tm, '    _nodes = _net[0]->NumNodes( );', '    AeState().clock = &_time;\n    _nodes = _net[0]->NumNodes( );')
    code['trafficmanager.cpp'] = tm
    code['gputrafficmanager.hpp'] = once(code['gputrafficmanager.hpp'], 'class GPUTrafficManager : public TrafficManager {',
                                        'class InterconnectInterface;\nclass GPUTrafficManager : public TrafficManager {\n  InterconnectInterface* _ae_interface = nullptr;')
    code['gputrafficmanager.cpp'] = code['gputrafficmanager.cpp'].replace('g_icnt_interface->', '_ae_interface->')

    # Routing identity & VC formula remain exactly the original dest_tag_fly.
    _, _, init = block(raw['routefunc.cpp'], 'void InitializeRoutingMap(')
    init = init[:init.index('  /* Register routing functions here */')]
    init += '\n  state.gRoutingFunctionMap["dest_tag_fly"] = &dest_tag_fly;\n'
    route = function(raw['routefunc.cpp'], 'void dest_tag_fly(')
    route = route.replace('void dest_tag_fly(', 'void dest_tag_fly(AeBookSimState& state, ')
    route = bind_globals(route, 'state')
    init = bind_globals(init.replace('state.gRoutingFunctionMap', 'gRoutingFunctionMap'), 'state')
    code['routefunc.cpp'] = '#include "routefunc.hpp"\n#include "router.hpp"\n#include "flit.hpp"\n#include "outputset.hpp"\n#include "config_utils.hpp"\n' + route + '\nvoid InitializeRoutingMap(const Configuration& config) {\n  auto& state=config.AeState();\n' + init + '}\n'
    code['routefunc.hpp'] = '''#ifndef AE_ROUTE_HPP
#define AE_ROUTE_HPP
#include "ae_state.hpp"
class Configuration;
void InitializeRoutingMap(const Configuration&);
#endif
'''
    for name in ('trafficmanager.cpp','gputrafficmanager.cpp','routers/iq_router.cpp'):
        code[name] = re.sub(r'\b_rf\(', '_rf(AeState(), ', code[name])
    code['vc.cpp'] = code['vc.cpp'].replace('  rf( router,', '  rf(AeState(), router,')

    # Pools belong to the network; Free uses the owner retained by that object.
    for klass, base in [('Flit','flit'),('Credit','credit'),('PacketReplyInfo','reply')]:
        file = {'reply':'packet_reply_info'}.get(base,base)
        h, c = code[file+'.hpp'], code[file+'.cpp']
        h = '#include "ae_state.hpp"\n' + h
        h = re.sub(r'static stack<'+klass+r'\s*\*>\s*_(?:all|free);', '', h)
        h = re.sub(r'(static\s+'+klass+r'\s*\*\s*New)\(\s*\)', r'\1(AeBookSimState& state)', h)
        h = h.replace('static void FreeAll();', 'static void FreeAll(AeBookSimState& state);')
        h = re.sub(r'(static int OutStanding)\(\s*\)', r'\1(AeBookSimState& state)', h)
        h = once(h, 'class '+klass+' {', 'class '+klass+' {\n  AeBookSimState* _ae_pool = nullptr;')
        c = re.sub(r'stack<'+klass+r'\s*\*>\s*'+klass+r'::_\w+;', '', c)
        c = re.sub(klass+r'::(New|FreeAll|OutStanding)\(\s*\)', klass+r'::\1(AeBookSimState& state)', c)
        _, _, free = block(c, 'void '+klass+'::Free(')
        c = body(c, 'void '+klass+'::Free(', free.replace('_free', '_ae_pool->'+base+'_free'))
        # Remaining unqualified pools occur only in static methods.
        c = re.sub(r'\b_all\b', 'state.'+base+'_all', c)
        c = re.sub(r'\b_free\b', 'state.'+base+'_free', c)
        var = {'Flit':'f','Credit':'c','PacketReplyInfo':'pr'}[klass]
        c = once(c, '  return '+var+';', '  '+var+'->_ae_pool=&state;\n  return '+var+';')
        c = once(c, 'void '+klass+'::FreeAll(AeBookSimState& state)', 'void '+klass+'::FreeAll(AeBookSimState& state)')
        a,b,part = block(c, 'void '+klass+'::FreeAll(')
        c = c[:a+1] + '\n  while(!state.'+base+'_free.empty()) state.'+base+'_free.pop();\n' + part + c[b-1:]
        code[file+'.hpp'], code[file+'.cpp'] = h,c
    for name in ('trafficmanager.cpp','gputrafficmanager.cpp','routers/iq_router.cpp'):
        for klass in ('Flit','Credit','PacketReplyInfo'):
            code[name] = re.sub(klass+r'::(New|FreeAll|OutStanding)\(\s*\)', klass+r'::\1(AeState())', code[name])
    code['ae_state.cpp'] = '''#include "ae_state.hpp"
#include "flit.hpp"
#include "credit.hpp"
#include "packet_reply_info.hpp"
AeBookSimState::~AeBookSimState() { PacketReplyInfo::FreeAll(*this); Flit::FreeAll(*this); Credit::FreeAll(*this); }
'''

    # External-traffic path constructs the original uniform/Bernoulli helpers.
    # They retain explicit owner references even though GPUTrafficManager never
    # uses them to inject: no unreachable global RNG remains in the library.
    code['traffic.hpp'] = '''#include "config_utils.hpp"
class TrafficPattern {
protected:
  int _nodes; AeBookSimState& state;
  TrafficPattern(int nodes, AeBookSimState& owner):_nodes(nodes),state(owner) {}
public:
  virtual ~TrafficPattern() {} virtual void reset() {}
  virtual int dest(int source)=0;
  static TrafficPattern* New(const string&,int,const Configuration*);
};
class UniformRandomTrafficPattern : public TrafficPattern {
public:
  UniformRandomTrafficPattern(int nodes,AeBookSimState& state):TrafficPattern(nodes,state) {}
  int dest(int source);
};
'''
    dest = function(raw['traffic.cpp'], 'int UniformRandomTrafficPattern::dest(').replace('RandomInt(', 'RandomInt(state, ')
    code['traffic.cpp'] = '#include "traffic.hpp"\n#include "random_utils.hpp"\n' + dest + '''
TrafficPattern* TrafficPattern::New(const string& pattern,int nodes,const Configuration* config) {
  if(pattern!="uniform" || !config) throw std::invalid_argument("unsupported synthetic traffic helper");
  return new UniformRandomTrafficPattern(nodes,config->AeState());
}
'''
    code['injection.hpp'] = '''#include "config_utils.hpp"
class InjectionProcess {
protected:
  int _nodes; double _rate; AeBookSimState& state;
  InjectionProcess(int nodes,double rate,AeBookSimState& owner):_nodes(nodes),_rate(rate),state(owner) {}
public:
  virtual ~InjectionProcess() {} virtual void reset() {} virtual bool test(int)=0;
  static InjectionProcess* New(const string&,int,double,const Configuration*);
};
class BernoulliInjectionProcess : public InjectionProcess {
public:
  BernoulliInjectionProcess(int n,double r,AeBookSimState& s):InjectionProcess(n,r,s) {}
  bool test(int source);
};
'''
    test = function(raw['injection.cpp'], 'bool BernoulliInjectionProcess::test(').replace('RandomFloat()', 'RandomFloat(state)')
    code['injection.cpp'] = '#include "injection.hpp"\n#include "random_utils.hpp"\n' + test + '''
InjectionProcess* InjectionProcess::New(const string& name,int nodes,double load,const Configuration* config) {
  if(name!="bernoulli" || !config || nodes<=0 || !(load>=0.0 && load<=1.0)) throw std::invalid_argument("unsupported injection helper");
  return new BernoulliInjectionProcess(nodes,load,config->AeState());
}
'''
    random = raw['random_utils.hpp']
    start = random.index(' // interface to Knuth')
    random = random[:start] + '''inline void RandomSeed(AeBookSimState& s,long seed) { s.ran_start(seed); s.ranf_start(seed); }
inline unsigned long RandomIntLong(AeBookSimState& s) { return s.next_int(); }
inline int RandomInt(AeBookSimState& s,int max) { return s.next_int() % (max+1); }
inline double RandomFloat(AeBookSimState& s) { return s.next_float(); }
inline double RandomFloat(AeBookSimState& s,double max) { return s.next_float()*max; }
#endif
'''
    code['random_utils.hpp'] = '#include "ae_state.hpp"\n' + random
    for name in ('trafficmanager.cpp','gputrafficmanager.cpp','allocators/pim.cpp'):
        for fn in ('RandomSeed','RandomInt','RandomIntLong','RandomFloat'):
            code[name] = re.sub(r'\b'+fn+r'\(\s*\)', fn+'(AeState())', code[name])
            code[name] = re.sub(r'\b'+fn+r'\((?!AeState\()', fn+'(AeState(), ', code[name])

    # Explicit typed injection replaces only the old mem_fetch type switch.
    h = code['interconnect_interface.hpp']
    h = '#include "ae_state.hpp"\n#include "flit.hpp"\n' + h
    h = constructor_declaration(h, 'InterconnectInterface', 'AeBookSimState& state')
    h = once(h, 'New(const char* const config_file)', 'New(const char* const config_file, AeBookSimState& state)')
    h = once(h, 'void* data, unsigned int size);', 'void* data, unsigned int size, Flit::FlitType packet_type);')
    h = once(h, '  virtual void Advance();', '''  void* AePeek(unsigned output) const;
  void AeValidate(unsigned requests, unsigned memories) const;
  AeBookSimState& AeState() const { return *_ae_state; }
  virtual void Advance();''')
    h = once(h, '  protected:', '  protected:') if '  protected:' in h else h
    h = once(h, 'class InterconnectInterface {', 'class InterconnectInterface {\n  AeBookSimState* _ae_state;')
    h = once(h, 'inline bool HasPacket() const { return _packet_n; }', 'inline bool HasPacket() const { return _packet_n; }\n    void* AePeek() const { return _packet_n ? _buffer.front() : nullptr; }')
    code['interconnect_interface.hpp'] = h
    c = code['interconnect_interface.cpp']
    for inc in ('mem_fetch.h','trace.h','power_module.hpp'):
        c = c.replace('#include "'+inc+'"', '')
    c = once(c, 'InterconnectInterface::InterconnectInterface()', 'InterconnectInterface::InterconnectInterface(AeBookSimState& state) : _ae_state(&state), _traffic_manager(nullptr), _icnt_config(nullptr), _subnets(0)')
    c = once(c, 'InterconnectInterface::New(const char* const config_file)', 'InterconnectInterface::New(const char* const config_file, AeBookSimState& state)')
    # New owns a parsed config but allocates no network until caller validates.
    a,b,_ = block(c,'InterconnectInterface* InterconnectInterface::New(')
    signature = c[c.index('InterconnectInterface* InterconnectInterface::New('):a]
    c = c[:c.index('InterconnectInterface* InterconnectInterface::New(')] + signature + '''{
  std::unique_ptr<InterconnectInterface> result(new InterconnectInterface(state));
  result->_icnt_config = new IntersimConfig();
  result->_icnt_config->AeBindState(state);
  if(!config_file || !*config_file) throw std::invalid_argument("BookSim config path is required");
  result->_icnt_config->ParseFile(config_file);
  return result.release();
}''' + c[b:]
    c = body(c,'InterconnectInterface::~InterconnectInterface(', '''  for(auto* network : _net) delete network;
  delete _traffic_manager;
  delete _icnt_config;
  AeState().clock=nullptr;''')
    c = once(c, '  _n_shader = n_shader;', '  AeValidate(n_shader,n_mem);\n  _n_shader = n_shader;')
    c = once(c, '  _flit_size =', '  _traffic_manager->_ae_interface=this;\n  _flit_size =') if '  _flit_size =' in c else c
    c = once(c, 'void *data, unsigned int size)', 'void *data, unsigned int size, Flit::FlitType packet_type)')
    start = c.index('  //TODO: Remove mem_fetch')
    end = c.index('  //TODO: _include_queuing',start)
    c = c[:start] + c[end:]
    c = re.sub(r'^\s*DPRINTF\([^\n]*\);', '', c, flags=re.M)
    c += '''
void* InterconnectInterface::AePeek(unsigned device) const {
  int node=_node_map.at(device), subnet=device<_n_shader ? 1 : 0;
  int turn=_round_robin_turn[subnet][node];
  for(int vc=0;vc<_vcs;++vc) {
    if(void* data=_boundary_buffer[subnet][node][turn].AePeek()) return data;
    if(++turn==_vcs) turn=0;
  }
  return nullptr;
}
''' + VALIDATE
    code['interconnect_interface.cpp'] = '#include <memory>\n#include <set>\n#include <string>\n#include <stdexcept>\n#include <cstdint>\n' + c
    # Member access always follows this explicit native owner; no selector.
    for name in ('trafficmanager.cpp','gputrafficmanager.cpp','routers/iq_router.cpp','networks/network.cpp','networks/fly.cpp','interconnect_interface.cpp','vc.cpp','flitchannel.cpp','buffer_state.cpp','channel.hpp'):
        code[name] = bind_globals(code[name])
        code[name] = code[name].replace('GetSimTime()', 'this->GetSimTime()')
    # LocalInterconnect already owns all mechanism state. Own the referenced
    # configuration by value and add passive views; arbitration bodies intact.
    lh, lc = raw['local_interconnect.h'], raw['local_interconnect.cc']
    boundary = local_boundary_member(lh, lc)
    lh = once(lh, 'const inct_config& m_inct_config;', 'const inct_config m_inct_config;')
    if lh.count('  void* Pop(unsigned ouput_deviceID);') != 2:
        raise ValueError('expected LocalInterconnect and xbar Pop declarations')
    lh = lh.replace('  void* Pop(unsigned ouput_deviceID);', '''  void* Pop(unsigned ouput_deviceID);
  void* AePeek(unsigned output) const { return out_buffers[output].empty() ? nullptr : out_buffers[output].front().data; }''', 1)
    lh = once(lh, '  bool HasPacket(unsigned deviceID) const;', '''  bool HasPacket(unsigned deviceID) const;
  void* AePeek(unsigned device) const { return net[device<''' + boundary + ''' ? 1 : 0]->AePeek(device); }
  bool AeHasBuffer(unsigned device) const { return net[device<''' + boundary + ''' ? 0 : 1]->Has_Buffer_In(device,1,false); }
  const vector<xbar_router*>& AeRouters() const { return net; }''')
    lc = lc.replace('#include "mem_fetch.h"','')
    code['local_interconnect.h'],code['local_interconnect.cc'] = lh,lc
    for name in ('routefunc.cpp','routefunc.hpp','traffic.cpp','traffic.hpp',
                 'injection.cpp','injection.hpp','globals.hpp'):
        # Preserve the upstream redistribution notice on extracted subsets.
        prefix = re.match(r'(?:(?:/\*.*?\*/|//[^\n]*|\s)+)', raw[name], re.S)
        if prefix:
            code[name] = prefix[0] + code[name]
    return close_base_declarations(code)


def generate(source_root: Path, output: Path | None = None):
    source_root = source_root.resolve()
    original = source_root / 'intersim2'
    output = output or original / 'ae_standalone'
    output = output.resolve()
    if output == original or output == source_root:
        raise ValueError('a separate generated output directory is required')
    if output.exists():
        raise FileExistsError(f'refusing to overwrite generated source: {output}')
    paths = sorted(p for p in original.rglob('*') if p.suffix in ('.hpp','.cpp','.c'))
    h = hashlib.sha256()
    raw, inputs = {}, {}
    for p in paths:
        name = p.relative_to(original).as_posix(); data=p.read_bytes()
        h.update(name.encode()+b'\0'+data+b'\0')
        raw[name]=data.decode(); inputs['intersim2/'+name]=sha(data)
    if h.hexdigest()!=TREE_SHA256:
        raise ValueError('pinned typed-v1 BookSim input tree mismatch; run before peek/extract')
    for name, expected in LOCAL_SHA256.items():
        data=(source_root/'gpgpu-sim'/name).read_bytes()
        if sha(data)!=expected: raise ValueError('pinned native LocalInterconnect mismatch: '+name)
        raw[name]=data.decode(); inputs['gpgpu-sim/'+name]=expected
    code=transform(raw)
    code['ae_state.hpp']=native_state(raw)
    names = [
        'module.cpp','config_utils.cpp','booksim_config.cpp','intersim_config.cpp',
        'buffer.cpp','vc.cpp','buffer_state.cpp','stats.cpp','flit.cpp','credit.cpp',
        'packet_reply_info.cpp','flitchannel.cpp','outputset.cpp','misc_utils.cpp',
        'routefunc.cpp','traffic.cpp','injection.cpp','trafficmanager.cpp',
        'gputrafficmanager.cpp','interconnect_interface.cpp','ae_state.cpp',
        'networks/network.cpp','networks/fly.cpp','routers/router.cpp','routers/iq_router.cpp',
        'power/buffer_monitor.cpp','power/switch_monitor.cpp','local_interconnect.cc',
    ]
    names += sorted(n for n in raw if re.fullmatch(r'(?:allocators|arbiters)/[^/]+\.cpp', n))
    # Compute transitive include closure. Unsupported topologies/types are not
    # linked merely because their sources exist next to the selected mechanism.
    needed=set(names); todo=list(names)
    basename={Path(n).name:n for n in code}
    while todo:
        name=todo.pop()
        for inc in re.findall(r'^\s*#\s*include\s+"([^"]+)"',code[name],re.M):
            dep=inc if inc in code else basename.get(inc)
            if dep is None: raise ValueError(f'non-BookSim dependency: {name}: {inc}')
            if dep not in needed: needed.add(dep); todo.append(dep)
    rendered = {}
    standard_includes = {}
    for name in sorted(needed):
        declared, required = direct_standard_declarations(code[name])
        rendered[name] = namespace(declared)
        if required:
            standard_includes[name] = required
    # Guards must encompass includes as well: BookSim has legitimate cyclic
    # header dependencies, and include hoisting must not bypass their guard.
    for name in rendered:
        if Path(name).suffix in ('.hpp','.h'):
            guard='AE_ONCE_'+re.sub(r'[^A-Za-z0-9]', '_', name).upper()
            rendered[name]='#ifndef '+guard+'\n#define '+guard+'\n'+rendered[name]+'\n#endif\n'
    special_members = special_member_signatures(rendered['interconnect_interface.hpp'],
                                                rendered['interconnect_interface.cpp'],
                                                'InterconnectInterface')
    destructors = audit_destructors(rendered)
    policy_sha = audit_validation_policy(rendered['interconnect_interface.cpp'])
    boundary = local_boundary_member(rendered['local_interconnect.h'], rendered['local_interconnect.cc'])
    _, _, local_header = block(rendered['local_interconnect.h'], 'class LocalInterconnect {')
    for helper in ('AePeek', 'AeHasBuffer'):
        declaration = ('void* ' if helper == 'AePeek' else 'bool ') + helper + '('
        _, _, native_helper = block(local_header, declaration)
        if not re.search(r'\bdevice\s*<\s*' + re.escape(boundary) + r'\b', cpp_code(native_helper)):
            raise ValueError('LocalInterconnect passive helper must use the real endpoint member')
    for name, headers in standard_includes.items():
        for header in headers:
            if not re.search(r'^\s*#\s*include\s*<' + header + '>', cpp_code(rendered[name]), re.M):
                raise ValueError(f'missing direct standard declaration {header}: {name}')
    manifest={
        'schema':'arch-explorer.standalone-icnt-source.v1',
        'input_tree_sha256':TREE_SHA256, 'inputs':inputs,
        'namespace':NS, 'sources':names,
        'include_dirs':['.','allocators','arbiters','routers','networks','power'],
        'flags':['-std=c++17'],
        'special_members':special_members, 'destructor_audit':destructors,
        'standard_includes':standard_includes, 'local_endpoint_boundary':boundary,
        'validation_policy':VALIDATION_POLICY, 'validation_policy_cpp_sha256':policy_sha,
        'outputs':{n:sha(t.encode()) for n,t in rendered.items()},
        'unsupported':['non-fly topology','non-dest_tag routing','non-iq router','non-islip allocator','nonidentity node mapping','synthetic injection','diagnostic file writers','NOQ/speculation/fault/power'],
    }
    # No writes occur before every input, anchor and dependency has validated.
    for name,text in rendered.items():
        p=output/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(text)
    (output/'SOURCES.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True,help='fresh typed-v1 gpgpusim/src')
    p.add_argument('--output',type=Path,help='new empty output directory')
    args=p.parse_args(); result=generate(args.source_root,args.output)
    print(json.dumps({'sources':len(result['sources']),'outputs':len(result['outputs']),'input_tree_sha256':result['input_tree_sha256']},sort_keys=True))
