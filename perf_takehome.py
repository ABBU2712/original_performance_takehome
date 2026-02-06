"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Dependency-aware slot packing for simple VLIW scheduling.
        if not vliw:
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs

        def addr_range(base, length):
            return set(range(base, base + length))

        def slot_rw(engine, slot):
            reads: set[int] = set()
            writes: set[int] = set()
            barrier = False

            if engine == "debug":
                # Keep debug comparisons in their own cycle to preserve ordering.
                op = slot[0]
                if op in ("compare", "vcompare"):
                    loc = slot[1]
                    reads.add(loc)
                barrier = True
                return reads, writes, barrier

            if engine == "alu":
                op, dest, a1, a2 = slot
                reads.update([a1, a2])
                writes.add(dest)
                return reads, writes, barrier

            if engine == "valu":
                match slot:
                    case ("vbroadcast", dest, src):
                        reads.add(src)
                        writes.update(addr_range(dest, VLEN))
                    case ("multiply_add", dest, a, b, c):
                        reads.update(addr_range(a, VLEN))
                        reads.update(addr_range(b, VLEN))
                        reads.update(addr_range(c, VLEN))
                        writes.update(addr_range(dest, VLEN))
                    case (op, dest, a1, a2):
                        reads.update(addr_range(a1, VLEN))
                        reads.update(addr_range(a2, VLEN))
                        writes.update(addr_range(dest, VLEN))
                    case _:
                        barrier = True
                return reads, writes, barrier

            if engine == "load":
                match slot:
                    case ("load", dest, addr):
                        reads.add(addr)
                        writes.add(dest)
                    case ("load_offset", dest, addr, offset):
                        reads.add(addr + offset)
                        writes.add(dest + offset)
                    case ("vload", dest, addr):
                        reads.add(addr)
                        writes.update(addr_range(dest, VLEN))
                    case ("const", dest, _val):
                        writes.add(dest)
                    case _:
                        barrier = True
                return reads, writes, barrier

            if engine == "store":
                match slot:
                    case ("store", addr, src):
                        reads.update([addr, src])
                    case ("vstore", addr, src):
                        reads.add(addr)
                        reads.update(addr_range(src, VLEN))
                    case _:
                        barrier = True
                return reads, writes, barrier

            if engine == "flow":
                match slot:
                    case ("select", dest, cond, a, b):
                        reads.update([cond, a, b])
                        writes.add(dest)
                    case ("vselect", dest, cond, a, b):
                        reads.update(addr_range(cond, VLEN))
                        reads.update(addr_range(a, VLEN))
                        reads.update(addr_range(b, VLEN))
                        writes.update(addr_range(dest, VLEN))
                    case ("add_imm", dest, a, _imm):
                        reads.add(a)
                        writes.add(dest)
                    case _:
                        barrier = True
                return reads, writes, barrier

            barrier = True
            return reads, writes, barrier

        instrs = []
        cur = {}
        cur_reads: set[int] = set()
        cur_writes: set[int] = set()

        def flush():
            nonlocal cur, cur_reads, cur_writes
            if cur:
                instrs.append(cur)
                cur = {}
                cur_reads = set()
                cur_writes = set()

        for engine, slot in slots:
            reads, writes, barrier = slot_rw(engine, slot)
            if barrier:
                flush()
                instrs.append({engine: [slot]})
                continue

            if len(cur.get(engine, [])) >= SLOT_LIMITS[engine]:
                flush()

            if reads & cur_writes:    #Relaxed
                flush()

            if len(cur.get(engine, [])) >= SLOT_LIMITS[engine]:
                flush()

            cur.setdefault(engine, []).append(slot)
            cur_reads.update(reads)
            cur_writes.update(writes)

        flush()
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        Vectorized SIMD kernel with:
        - idx/val loaded once per block and stored once per block
        - two-block software pipelining with interleaving to overlap load_offset (gather) with valu compute
        - hash stage fusion using multiply_add for ("+","+", "<<") pattern
        - parity update via (val & 1)
        - wrap via flow vselect
        """

        # -------------------------
        # Init scratch + params
        # -------------------------
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        def vec_const(val: int, name: str):
            vec_addr = self.alloc_scratch(name, VLEN)
            scalar = self.scratch_const(val)
            self.add("valu", ("vbroadcast", vec_addr, scalar))
            return vec_addr

        zero_v = vec_const(0, "zero_v")
        one_v = vec_const(1, "one_v")
        two_v = vec_const(2, "two_v")
        three_v = vec_const(3, "three_v")
        four_v = vec_const(4, "four_v")
        five_v = vec_const(5, "five_v")
        six_v = vec_const(6, "six_v")

        n_nodes_v = self.alloc_scratch("n_nodes_v", VLEN)
        self.add("valu", ("vbroadcast", n_nodes_v, self.scratch["n_nodes"]))

        forest_base_v = self.alloc_scratch("forest_base_v", VLEN)
        self.add("valu", ("vbroadcast", forest_base_v, self.scratch["forest_values_p"]))

        # Pre-broadcast hash constants + precompute mul constants for fused stages
        hash_const_v = {}
        hash_mul_v = {}
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            if val1 not in hash_const_v:
                hash_const_v[val1] = vec_const(val1, f"c_{val1}")
            if val3 not in hash_const_v:
                hash_const_v[val3] = vec_const(val3, f"c_{val3}")
            # If stage is: t1 = val + c1; t2 = val << s; val = t1 + t2
            # then val = val * (1 + 2^s) + c1  (mod 2^32)
            if (op1, op2, op3) == ("+", "+", "<<"):
                mul = (1 + (1 << val3)) % (2**32)
                if mul not in hash_mul_v:
                    hash_mul_v[mul] = vec_const(mul, f"mul_{mul}")

        # Required pause alignment for debug harness
        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        body = []

        # -------------------------
        # Scalar scratch
        # -------------------------
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        # Cache very top forest nodes for early rounds to avoid gather loads.
        top_node_v = []
        for node_i in range(7):  # nodes 0..6 cover rounds 0,1,2.
            node_s = self.alloc_scratch(f"top_node{node_i}_s")
            node_v = self.alloc_scratch(f"top_node{node_i}_v", VLEN)
            if node_i == 0:
                self.add("load", ("load", node_s, self.scratch["forest_values_p"]))
            else:
                self.add(
                    "alu",
                    (
                        "+",
                        tmp_addr,
                        self.scratch["forest_values_p"],
                        self.scratch_const(node_i),
                    ),
                )
                self.add("load", ("load", node_s, tmp_addr))
            self.add("valu", ("vbroadcast", node_v, node_s))
            top_node_v.append(node_v)

        # -------------------------
        # Vector scratch for 4-step pipeline
        # idx = [idx0, idx1, idx2, idx3]
        # val = [val0, val1, val2, val3]
        # node = [node0, node1, node2, node3]
        # addr = [addr0, addr1, addr2, addr3]
        # t1 = [t10, t11, t12, t13]
        # t2 = [t20, t21, t22, t23]
        # -------------------------
        idx_regs = [self.alloc_scratch(f"idx{i}", VLEN) for i in range(8)]
        val_regs = [self.alloc_scratch(f"val{i}", VLEN) for i in range(8)]
        node_regs = [self.alloc_scratch(f"node{i}", VLEN) for i in range(8)]
        addr_regs = [self.alloc_scratch(f"addr{i}", VLEN) for i in range(8)]
        t1_regs = [self.alloc_scratch(f"t1{i}", VLEN) for i in range(8)]
        t2_regs = [self.alloc_scratch(f"t2{i}", VLEN) for i in range(8)]

        # -------------------------
        # Helpers: return op lists
        # -------------------------
        def gather_addr_op(idx_reg, addr_reg):
            # addr_reg = forest_base_v + idx_reg
            return ("valu", ("+", addr_reg, idx_reg, forest_base_v))

        def gather_load_ops(node_reg, addr_reg):
            ops = []
            for lane in range(VLEN):
                ops.append(("load", ("load_offset", node_reg, addr_reg, lane)))
            return ops

        def gather_round_ops(idx_reg, node_reg, addr_reg, mask_reg, round_i):
            # Round 0 always reads forest[0], round 1 reads forest[1 or 2].
            if round_i == 0:
                return [("valu", ("+", node_reg, top_node_v[0], zero_v))]
            if round_i == 1:
                return [
                    ("valu", ("==", mask_reg, idx_reg, one_v)),
                    ("flow", ("vselect", node_reg, mask_reg, top_node_v[1], top_node_v[2])),
                ]
            if round_i == 2:
                ops = []
                # Default to node6, then override with vselects for 5,4,3.
                ops.append(("valu", ("+", node_reg, top_node_v[6], zero_v)))
                ops.append(("valu", ("==", mask_reg, idx_reg, five_v)))
                ops.append(("flow", ("vselect", node_reg, mask_reg, top_node_v[5], node_reg)))
                ops.append(("valu", ("==", mask_reg, idx_reg, four_v)))
                ops.append(("flow", ("vselect", node_reg, mask_reg, top_node_v[4], node_reg)))
                ops.append(("valu", ("==", mask_reg, idx_reg, three_v)))
                ops.append(("flow", ("vselect", node_reg, mask_reg, top_node_v[3], node_reg)))
                return ops
            ops = [gather_addr_op(idx_reg, addr_reg)]
            ops.extend(gather_load_ops(node_reg, addr_reg))
            return ops

        def compute_ops_split(idx_reg, val_reg, node_reg, t1_reg, t2_reg, update_idx=True):
            valu_ops = []
            flow_ops = []
            # val = hash(val ^ node)
            valu_ops.append(("valu", ("^", val_reg, val_reg, node_reg)))
            for op1, val1, op2, op3, val3 in HASH_STAGES:
                if (op1, op2, op3) == ("+", "+", "<<"):
                    mul = (1 + (1 << val3)) % (2**32)
                    valu_ops.append(
                        (
                            "valu",
                            (
                                "multiply_add",
                                val_reg,               # out
                                val_reg,               # a
                                hash_mul_v[mul],        # mul
                                hash_const_v[val1],     # add
                            ),
                        )
                    )
                else:
                    valu_ops.append(("valu", (op1, t1_reg, val_reg, hash_const_v[val1])))
                    valu_ops.append(("valu", (op3, t2_reg, val_reg, hash_const_v[val3])))
                    valu_ops.append(("valu", (op2, val_reg, t1_reg, t2_reg)))

            if update_idx:
                # idx = 2*idx + 1 + (val & 1)
                valu_ops.append(("valu", ("&", t1_reg, val_reg, one_v)))        # t1 = val & 1
                valu_ops.append(("valu", ("+", t1_reg, t1_reg, one_v)))         # t1 = t1 + 1
                valu_ops.append(("valu", ("multiply_add", idx_reg, idx_reg, two_v, t1_reg)))  # idx = idx*2 + t1

                # wrap: idx = (idx < n_nodes) ? idx : 0  (use valu mask to avoid flow)
                valu_ops.append(("valu", ("<", t2_reg, idx_reg, n_nodes_v)))    # t2 = mask
                valu_ops.append(("valu", ("*", idx_reg, idx_reg, t2_reg)))
            return valu_ops, flow_ops

        def round_robin_ops(op_lists):
            """
            Interleave ops from multiple independent stages in round-robin order.
            Preserves per-stage order while maximizing VLIW packing across stages.
            """
            out = []
            remaining = True
            while remaining:
                remaining = False
                for ops in op_lists:
                    if ops:
                        out.append(ops.pop(0))
                        remaining = True
            return out


        # -------------------------
        # Main vector blocks: 8-step software pipeline
        # -------------------------
        vec_blocks = batch_size // VLEN
        vec_end = vec_blocks * VLEN

        bi = 0
        while bi + 7 < vec_blocks:
            base_consts = [self.scratch_const((bi + k) * VLEN) for k in range(8)]

            # Load idx/val once for 8 blocks
            for k in range(8):
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], base_consts[k])))
                body.append(("load", ("vload", idx_regs[k], tmp_addr)))
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], base_consts[k])))
                body.append(("load", ("vload", val_regs[k], tmp_addr)))

            for r in range(rounds):
                # Gather nodes for all 8 stages first (ensures node regs are ready).
                for stage in range(8):
                    body.extend(
                        gather_round_ops(
                            idx_regs[stage],
                            node_regs[stage],
                            addr_regs[stage],
                            t1_regs[stage],
                            r,
                        )
                    )

                # Interleave valu ops across stages to maximize valu slot utilization.
                valu_lists = []
                flow_lists = []
                for stage in range(8):
                    valu_ops, flow_ops = compute_ops_split(
                        idx_regs[stage],
                        val_regs[stage],
                        node_regs[stage],
                        t1_regs[stage],
                        t2_regs[stage],
                        update_idx=(r != rounds - 1),
                    )
                    valu_lists.append(valu_ops)
                    flow_lists.append(flow_ops)
                body.extend(round_robin_ops(valu_lists))
                body.extend(round_robin_ops(flow_lists))

            # Store values once for all 8 blocks
            for k in range(8):
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], base_consts[k])))
                body.append(("store", ("vstore", tmp_addr, val_regs[k])))

            bi += 8

        # -------------------------
        # Leftover vector blocks (< 8)
        # -------------------------
        while bi < vec_blocks:
            base_c = self.scratch_const(bi * VLEN)

            body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], base_c)))
            body.append(("load", ("vload", idx_regs[0], tmp_addr)))
            body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], base_c)))
            body.append(("load", ("vload", val_regs[0], tmp_addr)))

            body.extend(
                gather_round_ops(idx_regs[0], node_regs[0], addr_regs[0], t1_regs[0], 0)
            )

            for r in range(rounds):
                valu_ops, flow_ops = compute_ops_split(
                    idx_regs[0],
                    val_regs[0],
                    node_regs[0],
                    t1_regs[0],
                    t2_regs[0],
                    update_idx=(r != rounds - 1),
                )
                body.extend(valu_ops)
                body.extend(flow_ops)
                if r != rounds - 1:
                    body.extend(
                        gather_round_ops(
                            idx_regs[0], node_regs[0], addr_regs[0], t1_regs[0], r + 1
                        )
                    )

            body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], base_c)))
            body.append(("store", ("vstore", tmp_addr, val_regs[0])))

            bi += 1

        # -------------------------
        # Scalar tail (non-multiple of VLEN)
        # -------------------------
        for i in range(vec_end, batch_size):
            i_c = self.scratch_const(i)

            # load idx/val once
            body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_c)))
            body.append(("load", ("load", tmp_idx, tmp_addr)))
            body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_c)))
            body.append(("load", ("load", tmp_val, tmp_addr)))

            for r in range(rounds):
                # node = forest[idx]
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))

                # hash
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                for op1, val1, op2, op3, val3 in HASH_STAGES:
                    if (op1, op2, op3) == ("+", "+", "<<"):
                        mul_c = self.scratch_const((1 + (1 << val3)) % (2**32))
                        add_c = self.scratch_const(val1)
                        body.append(("alu", ("*", tmp1, tmp_val, mul_c)))
                        body.append(("alu", ("+", tmp_val, tmp1, add_c)))
                    else:
                        body.append(("alu", (op1, tmp1, tmp_val, self.scratch_const(val1))))
                        body.append(("alu", (op3, tmp2, tmp_val, self.scratch_const(val3))))
                        body.append(("alu", (op2, tmp_val, tmp1, tmp2)))

                if r != rounds - 1:
                    # idx update
                    body.append(("alu", ("&", tmp1, tmp_val, one_const)))   # tmp1 = val & 1
                    body.append(("alu", ("+", tmp1, tmp1, one_const)))      # tmp1 = tmp1 + 1
                    body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                    body.append(("alu", ("+", tmp_idx, tmp_idx, tmp1)))

                    # wrap idx
                    body.append(("alu", ("<", tmp2, tmp_idx, self.scratch["n_nodes"])))
                    body.append(("flow", ("select", tmp_idx, tmp2, tmp_idx, zero_const)))

            # store values once
            body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_c)))
            body.append(("store", ("store", tmp_addr, tmp_val)))

        # Build + emit
        body_instrs = self.build(body, vliw=True)
        self.instrs.extend(body_instrs)
        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
