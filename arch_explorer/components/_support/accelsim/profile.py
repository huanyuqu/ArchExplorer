# Concrete Accel-Sim implementation; owned by the component provider.
"""Fixed SM90_H100 profile, transcribed from the pinned upstream configurations.

Native options not exposed for exploration are fixed ADF parameters, not another
configuration file. ENGINE_DEFAULTS documents remaining upstream parser defaults.
Copyright for upstream parameter selections: see docs/accelsim.md; no code vendored.
"""
ACCELSIM_SHA = '1c9cc831c5d948f89fa9dc8f21f6c87952f5f2ea'
GPGPUSIM_SHA = 'c24e8858eee552b3bc2919b147f14e2a88571775'
PROFILE = 'sm90-h100-v2.0.1-subset-v1'
# owner, native key, default value, unit, source location
FIELDS = (
    ("sm", "gpgpu_ptx_instruction_classification", 0, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:2"),
    ("sm", "gpgpu_ptx_sim_mode", 0, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:3"),
    ("sm", "gpgpu_ptx_force_max_capability", 90, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:4"),
    ("sm", "gpgpu_stack_size_limit", 1024, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:7"),
    ("sm", "gpgpu_heap_size_limit", 8388608, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:8"),
    ("sm", "gpgpu_runtime_sync_depth_limit", 2, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:9"),
    ("sm", "gpgpu_runtime_pending_launch_count_limit", 2048, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:10"),
    ("sm", "gpgpu_kernel_launch_latency", 5000, "cycle", "configs/tested-cfgs/SM90_H100/gpgpusim.config:11"),
    ("sm", "gpgpu_TB_launch_latency", 0, "cycle", "configs/tested-cfgs/SM90_H100/gpgpusim.config:12"),
    ("sm", "gpgpu_max_concurrent_kernel", 128, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:13"),
    ("sm", "gpgpu_compute_capability_major", 9, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:16"),
    ("sm", "gpgpu_compute_capability_minor", 0, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:17"),
    ("sm", "gpgpu_ptx_convert_to_ptxplus", 0, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:20"),
    ("sm", "gpgpu_ptx_save_converted_ptxplus", 0, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:21"),
    ("sm", "gpgpu_n_cores_per_cluster", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:25"),
    ("sm", "gpgpu_clock_gated_lanes", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:28"),
    ("dram", "gpgpu_n_chiplet_partition", 2, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:29"),
    ("dram", "inter_chiplet_queue_size", 256, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:30"),
    ("dram", "gpgpu_chiplet_interleave", 0, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:31"),
    ("sm", "gpgpu_shader_registers", 65536, "register", "configs/tested-cfgs/SM90_H100/gpgpusim.config:38"),
    ("sm", "gpgpu_registers_per_block", 65536, "register", "configs/tested-cfgs/SM90_H100/gpgpusim.config:39"),
    ("sm", "gpgpu_occupancy_sm_number", 90, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:40"),
    ("sm", "gpgpu_shader_core_pipeline", "2048:32", "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:42"),
    ("sm", "gpgpu_shader_cta", 32, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:43"),
    ("sm", "gpgpu_simd_model", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:44"),
    ("sm", "gpgpu_pipeline_widths", "4,4,4,4,4,4,4,4,4,4,8,4,4", "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:48"),
    ("sm", "gpgpu_num_sp_units", 4, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:49"),
    ("sm", "gpgpu_num_sfu_units", 4, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:50"),
    ("sm", "gpgpu_num_dp_units", 4, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:51"),
    ("sm", "gpgpu_num_int_units", 4, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:52"),
    ("sm", "gpgpu_tensor_core_avail", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:53"),
    ("sm", "gpgpu_num_tensor_core_units", 4, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:54"),
    ("sm", "ptx_opcode_latency_int", "4,4,4,4,21", "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:59"),
    ("sm", "ptx_opcode_initiation_int", "2,2,2,2,2", "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:60"),
    ("sm", "ptx_opcode_latency_fp", "4,4,4,4,39", "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:61"),
    ("sm", "ptx_opcode_initiation_fp", "2,2,2,2,4", "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:62"),
    ("sm", "ptx_opcode_latency_dp", "8,8,8,8,330", "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:63"),
    ("sm", "ptx_opcode_initiation_dp", "2,2,2,2,130", "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:64"),
    ("sm", "ptx_opcode_latency_sfu", 23, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:65"),
    ("sm", "ptx_opcode_initiation_sfu", 8, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:66"),
    ("sm", "ptx_opcode_latency_tesnor", 24, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:67"),
    ("sm", "ptx_opcode_initiation_tensor", 8, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:68"),
    ("sm", "gpgpu_sub_core_model", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:72"),
    ("sm", "gpgpu_enable_specialized_operand_collector", 0, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:74"),
    ("sm", "gpgpu_operand_collector_num_units_gen", 8, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:75"),
    ("sm", "gpgpu_operand_collector_num_in_ports_gen", 8, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:76"),
    ("sm", "gpgpu_operand_collector_num_out_ports_gen", 8, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:77"),
    ("sm", "gpgpu_num_reg_banks", 64, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:79"),
    ("sm", "gpgpu_reg_file_port_throughput", 2, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:80"),
    ("sm", "gpgpu_num_sched_per_core", 4, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:83"),
    ("sm", "gpgpu_scheduler", "lrr", "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:84"),
    ("sm", "gpgpu_opndcoll_model", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:85"),
    ("sm", "gpgpu_max_insn_issue_per_warp", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:87"),
    ("sm", "gpgpu_dual_issue_diff_exec_units", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:88"),
    ("l1_shared", "gpgpu_adaptive_cache_config", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:95"),
    ("l1_shared", "gpgpu_shmem_option", "0,8,16,32,64,228", "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:96"),
    ("l1_shared", "gpgpu_unified_l1d_size", 256, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:97"),
    ("l1_shared", "gpgpu_l1_banks", 4, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:99"),
    ("l1_shared", "gpgpu_cache:dl1", "S:4:128:64,L:T:m:L:L,A:512:64,16:0,32", "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:100"),
    ("l1_shared", "gpgpu_l1_latency", 41, "cycle", "configs/tested-cfgs/SM90_H100/gpgpusim.config:101"),
    ("l1_shared", "gpgpu_gmem_skip_L1D", 0, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:102"),
    ("l1_shared", "gpgpu_flush_l1_cache", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:103"),
    ("icnt", "gpgpu_n_cluster_ejection_buffer_size", 32, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:104"),
    ("l1_shared", "gpgpu_l1_cache_write_ratio", 25, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:105"),
    ("l1_shared", "gpgpu_n_ldst_response_buffer_size", 32, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:106"),
    ("l1_shared", "gpgpu_shmem_size", 233472, "B", "configs/tested-cfgs/SM90_H100/gpgpusim.config:109"),
    ("l1_shared", "gpgpu_shmem_sizeDefault", 233472, "B", "configs/tested-cfgs/SM90_H100/gpgpusim.config:110"),
    ("l1_shared", "gpgpu_shmem_per_block", 49152, "B", "configs/tested-cfgs/SM90_H100/gpgpusim.config:111"),
    ("l1_shared", "gpgpu_smem_latency", 29, "cycle", "configs/tested-cfgs/SM90_H100/gpgpusim.config:112"),
    ("l1_shared", "gpgpu_shmem_num_banks", 32, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:114"),
    ("l1_shared", "gpgpu_shmem_limited_broadcast", 0, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:115"),
    ("l1_shared", "gpgpu_shmem_warp_parts", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:116"),
    ("l1_shared", "gpgpu_coalesce_arch", 90, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:117"),
    ("l2", "gpgpu_cache:dl2", "S:256:128:20,L:B:m:L:X,A:384:4,32:0,32", "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:120"),
    ("l2", "gpgpu_cache:dl2_texture_only", 0, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:121"),
    ("dram", "gpgpu_dram_partition_queues", "64:64:64:64", "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:122"),
    ("l2", "gpgpu_perf_sim_memcpy", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:123"),
    ("dram", "gpgpu_memory_partition_indexing", 6, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:125"),
    ("l2", "gpgpu_lrc_enabled", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:128"),
    ("l2", "gpgpu_lrc_max_entries", 1024, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:131"),
    ("l2", "gpgpu_lrc_max_merged", 4, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:132"),
    ("sm", "gpgpu_cache:il1", "N:64:128:16,L:R:f:N:L,S:2:48,4", "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:136"),
    ("sm", "gpgpu_inst_fetch_throughput", 4, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:137"),
    ("sm", "gpgpu_tex_cache:l1", "N:4:128:256,L:R:m:N:L,T:512:8,128:2", "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:140"),
    ("sm", "gpgpu_const_cache:l1", "N:128:64:8,L:R:f:N:L,S:2:64,4", "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:142"),
    ("sm", "gpgpu_perfect_inst_const_cache", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:143"),
    ("icnt", "network_mode", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:147"),
    ("icnt", "inter_config_file", "config_hopper_islip.icnt", "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:148"),
    ("l2", "gpgpu_l2_rop_latency", 237, "cycle", "configs/tested-cfgs/SM90_H100/gpgpusim.config:151"),
    ("dram", "dram_latency", 194, "cycle", "configs/tested-cfgs/SM90_H100/gpgpusim.config:152"),
    ("dram", "gpgpu_dram_scheduler", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:155"),
    ("dram", "gpgpu_frfcfs_dram_sched_queue_size", 128, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:156"),
    ("dram", "gpgpu_dram_return_queue_size", 384, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:157"),
    ("dram", "gpgpu_n_mem_per_ctrlr", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:160"),
    ("dram", "gpgpu_dram_buswidth", 8, "B", "configs/tested-cfgs/SM90_H100/gpgpusim.config:161"),
    ("dram", "gpgpu_dram_burst_length", 4, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:162"),
    ("dram", "dram_data_command_freq_ratio", 2, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:163"),
    ("dram", "gpgpu_mem_address_mask", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:164"),
    ("dram", "gpgpu_mem_addr_mapping", "dramid@8;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RBBBCCCC.BCCSSSSS", "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:165"),
    ("dram", "gpgpu_dram_timing_opt", "nbk=16:CCD=1:RRD=6:RCD=19:RAS=45:RP=19:RC=50:CL=19:WL=6:CDLR=8:WR=21:nbkgrp=4:CCDL=6:RTPL=8", "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:171"),
    ("dram", "dram_dual_bus_interface", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:172"),
    ("dram", "dram_bnk_indexing_policy", 0, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:175"),
    ("dram", "dram_bnkgrp_indexing_policy", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:176"),
    ("sm", "gpgpu_memlatency_stat", 14, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:182"),
    ("sm", "gpgpu_runtime_stat", 500, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:183"),
    ("sm", "enable_ptx_file_line_stats", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:184"),
    ("sm", "visualizer_enabled", 0, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:185"),
    ("sm", "power_simulation_enabled", 0, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:188"),
    ("dram", "gpgpu_simple_dram_model", 1, "1", "configs/tested-cfgs/SM90_H100/gpgpusim.config:196"),
    ("sm", "trace_opcode_latency_initiation_int", "4,2", "1", "gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config:1"),
    ("sm", "trace_opcode_latency_initiation_sp", "4,2", "1", "gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config:2"),
    ("sm", "trace_opcode_latency_initiation_dp", "8,2", "1", "gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config:3"),
    ("sm", "trace_opcode_latency_initiation_sfu", "23,8", "1", "gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config:4"),
    ("sm", "trace_opcode_latency_initiation_tensor", "12,4", "1", "gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config:5"),
    ("sm", "specialized_unit_1", "1,4,4,4,4,BRA", "1", "gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config:9"),
    ("sm", "trace_opcode_latency_initiation_spec_op_1", "4,4", "1", "gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config:10"),
    ("sm", "specialized_unit_2", "1,4,200,4,4,TEX", "1", "gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config:13"),
    ("sm", "trace_opcode_latency_initiation_spec_op_2", "200,4", "1", "gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config:14"),
    ("sm", "specialized_unit_3", "1,4,12,4,4,TENSOR", "1", "gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config:17"),
    ("sm", "trace_opcode_latency_initiation_spec_op_3", "12,4", "1", "gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config:18"),
    ("sm", "specialized_unit_4", "1,4,4,4,4,UDP", "1", "gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config:21"),
    ("sm", "trace_opcode_latency_initiation_spec_op_4", "4,1", "1", "gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config:22"),
    ("sm", "specialized_unit_5", "1,4,128,4,4,GMMA", "1", "gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config:29"),
    ("sm", "trace_opcode_latency_initiation_spec_op_5", "128,32", "1", "gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config:30"),
    ('icnt', "booksim_use_map", 0, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:2'),
    ('icnt', "booksim_flit_size", 40, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:3'),
    ('icnt', "booksim_network_count", 2, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:6'),
    ('icnt', "booksim_topology", "fly", '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:9'),
    ('icnt', "booksim_k", 212, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:10'),
    ('icnt', "booksim_n", 1, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:11'),
    ('icnt', "booksim_routing_function", "dest_tag", '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:15'),
    ('icnt', "booksim_num_vcs", 1, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:20'),
    ('icnt', "booksim_vc_buf_size", 256, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:21'),
    ('icnt', "booksim_input_buffer_size", 256, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:22'),
    ('icnt', "booksim_ejection_buffer_size", 256, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:23'),
    ('icnt', "booksim_boundary_buffer_size", 256, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:24'),
    ('icnt', "booksim_wait_for_tail_credit", 0, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:26'),
    ('icnt', "booksim_vc_allocator", "islip", '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:30'),
    ('icnt', "booksim_sw_allocator", "islip", '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:31'),
    ('icnt', "booksim_alloc_iters", 1, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:32'),
    ('icnt', "booksim_credit_delay", 0, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:34'),
    ('icnt', "booksim_routing_delay", 0, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:35'),
    ('icnt', "booksim_vc_alloc_delay", 1, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:36'),
    ('icnt', "booksim_sw_alloc_delay", 1, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:37'),
    ('icnt', "booksim_input_speedup", 1, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:39'),
    ('icnt', "booksim_output_speedup", 1, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:40'),
    ('icnt', "booksim_internal_speedup", "4.0", '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:41'),
    ('icnt', "booksim_traffic", "uniform", '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:45'),
    ('icnt', "booksim_packet_size", "{{1,2,3,4},{10,20}}", '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:46'),
    ('icnt', "booksim_packet_size_rate", "{{1,1,1,1},{2,1}}", '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:47'),
    ('icnt', "booksim_sim_type", "gpgpusim", '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:51'),
    ('icnt', "booksim_injection_rate", "0.1", '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:53'),
    ('icnt', "booksim_subnets", 2, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:55'),
    ('icnt', "booksim_read_request_subnet", 0, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:61'),
    ('icnt', "booksim_read_reply_subnet", 1, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:62'),
    ('icnt', "booksim_write_request_subnet", 0, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:63'),
    ('icnt', "booksim_write_reply_subnet", 1, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:64'),
    ('icnt', "booksim_read_request_begin_vc", 0, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:66'),
    ('icnt', "booksim_read_request_end_vc", 0, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:67'),
    ('icnt', "booksim_write_request_begin_vc", 0, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:68'),
    ('icnt', "booksim_write_request_end_vc", 0, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:69'),
    ('icnt', "booksim_read_reply_begin_vc", 0, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:70'),
    ('icnt', "booksim_read_reply_end_vc", 0, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:71'),
    ('icnt', "booksim_write_reply_begin_vc", 0, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:72'),
    ('icnt', "booksim_write_reply_end_vc", 0, '1', 'configs/tested-cfgs/SM90_H100/config_hopper_islip.icnt:73'),
)

CHOICES = {
    'gpgpu_lrc_enabled': [0, 1],
    'gpgpu_shader_registers': [32768, 65536],
    'gpgpu_registers_per_block': [32768, 65536],
    'gpgpu_shader_cta': [16, 32],
    'gpgpu_l1_latency': [41, 42, 48],
    'gpgpu_smem_latency': [29, 30],
    'gpgpu_l2_rop_latency': [237, 238],
    'dram_latency': [194, 195, 200],
    'booksim_input_buffer_size': [128, 256, 512],
    'booksim_vc_buf_size': [128, 256, 512],
    'booksim_k': sorted({sm + 2 * mem for sm in [2, 4, 8, 16, 32, 64, 128, 132] for mem in [20, 40]}),
}
# Physical units for scalar fields; composite native strings carry their
# upstream grammar (cache sets/line bytes/ways/policies, pipelines or timings).
_UNITS = {
    'gpgpu_shader_cta': 'count', 'gpgpu_num_sched_per_core': 'count',
    'gpgpu_n_cores_per_cluster': 'count', 'gpgpu_n_chiplet_partition': 'count',
    'gpgpu_unified_l1d_size': 'KiB', 'gpgpu_shmem_option': 'KiB-list',
    'booksim_flit_size': 'B', 'booksim_input_buffer_size': 'flit',
    'booksim_vc_buf_size': 'flit', 'booksim_ejection_buffer_size': 'flit',
    'booksim_boundary_buffer_size': 'flit', 'booksim_k': 'count',
    'booksim_n': 'count', 'booksim_num_vcs': 'count',
}
FIELDS = tuple((owner, key, value, _UNITS.get(key, 'native-format' if isinstance(value, str) else unit), location)
               for owner, key, value, unit, location in FIELDS)
CLOCKS = {'sm': 1980, 'icnt': 3960, 'l2': 3960, 'dram': 3106}
OBJECTS = {'sm': 'trace_shader_core_ctx / scheduler_unit',
           'l1_shared': 'ldst_unit / L1D / shared memory',
           'icnt': 'InterconnectInterface / BookSim single-stage fly', 'l2': 'memory_sub_partition / l2_cache',
           'dram': 'memory_partition_unit / dram_t'}
ENGINE_DEFAULTS = 'Unlisted parser options remain pinned upstream defaults; no user-supplied config or flags.'

# Retained with the transcribed upstream configuration values in wheels/sdists.
THIRD_PARTY_NOTICES = """
Accel-Sim framework LICENSE
BSD 2-Clause License

Copyright (c) 2018-2021, Vijay Kandiah, Junrui Pan, Mahmoud Khairy, Scott Peverelle, Timothy Rogers, Tor M. Aamodt, Nikos Hardavellas
Northwestern University, Purdue University, The University of British Columbia
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer;
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution;
3. Neither the names of Northwestern University, Purdue University,
   The University of British Columbia nor the names of their contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

GPGPU-Sim COPYRIGHT
Copyright (c) 2009-2011 
Tor M. Aamodt and The University of British Columbia (UBC)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.
Neither the name of the copyright holders nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


NOTE 1: The files libcuda/cuda_runtime_api.cc and src/cuda-sim/cuda-math.h are
derived from the CUDA Toolset available from http://www.nvidia.com/cuda
(property of NVIDIA).  The files from src/intersim/ are derived from Booksim (a
simulator provided with the textbook "Principles and Practices of
Interconnection Networks" available from http://cva.stanford.edu/books/ppin/).
The files from src/cuda-sim/decuda_pred_table/ are derived from decuda 
(https://github.com/laanwj/decuda/wiki). 
As such, those files are bound by the corresponding legal terms and conditions
set forth separately (original copyright notices are left in files from these
sources and where we have modified a file our copyright notice appears before
the original copyright notice).

NOTE 2: Various UBC graduate students have contributed to different files in
this distribution.  As copyright ownership at UBC is held by the authors (as
per UBC policy 88, item 2.3 on literary works) these students names appear in
the copyright notices of the respective files. UBC is also mentioned in the 
copyright notice to highlight that was the author's affiliation when the work 
was performed.

NOTE 3: AccelWattch and all its components are covered by the following license and copyright.
Copyright (c) 2018-2021, Vijay Kandiah, Junrui Pan, Mahmoud Khairy, Scott Peverelle, Timothy Rogers, Tor M. Aamodt, Nikos Hardavellas
Northwestern University, Purdue University, The University of British Columbia
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer;
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution;
3. Neither the names of Northwestern University, Purdue University,
   The University of British Columbia nor the names of their contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

