"""Fixed upstream identity; the SIM38 borrowed-host builder is retired.

Frozen original-cycle manifests and their private build inputs remain valid
read-only evidence. New builds use standalone_build.py and public factories.
"""
BASES = {'accelsim':'1c9cc831c5d948f89fa9dc8f21f6c87952f5f2ea',
         'gpgpusim':'c24e8858eee552b3bc2919b147f14e2a88571775'}
PATCHES = {'accelsim':'f90e0be472a1e657e02b3a3de5c2f7217d8be8dcb169bf2eb46e41eefa37b477',
           'gpgpusim':'b250da0dfd4d69154f7280e4311cf4641ba14e969738568d839f64038b389891'}


def main():
    raise SystemExit('The borrowed-host SIM38 builder is archival. Use standalone_build.py for independent factories; consume the frozen original-cycle oracle separately.')


if __name__ == '__main__':
    main()
