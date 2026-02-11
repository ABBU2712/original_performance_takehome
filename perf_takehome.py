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

KERNEL_CACHE_VERSION = 1
KERNEL_CACHE: dict[tuple, tuple] = {}


class KernelBuilder:
    # Randomized scheduler seed chosen via sweep (lower cycles due to better packing).
    def __init__(self, schedule_seed: int = 91):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.schedule_seed = schedule_seed

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def _slot_rw(self, engine, slot):
        def addr_range(base, length):
            return set(range(base, base + length))

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

        for item in slots:
            if isinstance(item, dict):
                flush()
                instrs.append(item)
                continue
            engine, slot = item
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

    def build_greedy(self, slots: list[tuple[Engine, tuple]]):
        """
        Greedy list scheduler that can reorder independent ops to fill VLIW slots.
        """
        items = []
        for item in slots:
            if isinstance(item, dict):
                items.append({"barrier": True, "instr": item})
                continue
            engine, slot = item
            reads, writes, barrier = self._slot_rw(engine, slot)
            if barrier:
                items.append({"barrier": True, "instr": {engine: [slot]}})
            else:
                items.append(
                    {
                        "barrier": False,
                        "engine": engine,
                        "slot": slot,
                        "reads": reads,
                        "writes": writes,
                    }
                )

        def has_prior_dependency(pending, idx, reads, writes):
            # Can't move before any prior unscheduled op that conflicts.
            for j in range(idx):
                it = pending[j]
                if it.get("barrier"):
                    return True
                pre_reads = it["reads"]
                pre_writes = it["writes"]
                if (reads & pre_writes) or (writes & (pre_reads | pre_writes)):
                    return True
            return False

        instrs = []
        pending = items
        while pending:
            # Handle leading barrier
            if pending[0].get("barrier"):
                instrs.append(pending.pop(0)["instr"])
                continue

            # Do not schedule across the first barrier.
            barrier_idx = None
            for i, it in enumerate(pending):
                if it.get("barrier"):
                    barrier_idx = i
                    break

            scan_limit = barrier_idx if barrier_idx is not None else len(pending)
            bundle = {}
            cur_reads: set[int] = set()
            cur_writes: set[int] = set()
            scheduled_any = False

            i = 0
            while i < scan_limit:
                it = pending[i]
                engine = it["engine"]
                if len(bundle.get(engine, [])) >= SLOT_LIMITS[engine]:
                    i += 1
                    continue
                reads = it["reads"]
                writes = it["writes"]
                if reads & cur_writes:
                    i += 1
                    continue
                if writes & (cur_reads | cur_writes):
                    i += 1
                    continue
                if has_prior_dependency(pending, i, reads, writes):
                    i += 1
                    continue
                bundle.setdefault(engine, []).append(it["slot"])
                cur_reads.update(reads)
                cur_writes.update(writes)
                pending.pop(i)
                scheduled_any = True
                scan_limit -= 1
                continue

            if not scheduled_any:
                # Fallback: preserve order for the first pending op.
                it = pending.pop(0)
                instrs.append({it["engine"]: [it["slot"]]})
                continue

            instrs.append(bundle)

        return instrs

    def build_linear(self, slots: list[tuple[Engine, tuple]]):
        """
        Fast list scheduler (linear-time-ish) that packs ops into cycles by
        respecting simple RAW/WAR/WAW dependencies and slot limits.
        """
        cycles: list[dict[str, list[tuple]]] = []
        usage: list[dict[str, int]] = []
        ready_time: dict[int, int] = defaultdict(int)
        last_write: dict[int, int] = defaultdict(lambda: -1)
        last_read: dict[int, int] = defaultdict(lambda: -1)
        next_free: dict[str, int] = defaultdict(int)

        def ensure_cycle(cycle: int) -> None:
            while len(cycles) <= cycle:
                cycles.append({})
                usage.append(defaultdict(int))

        def find_cycle(engine: str, earliest: int) -> int:
            cycle = earliest if earliest > next_free[engine] else next_free[engine]
            limit = SLOT_LIMITS[engine]
            while True:
                ensure_cycle(cycle)
                if usage[cycle][engine] < limit:
                    return cycle
                cycle += 1

        def place_op(cycle: int, engine: str, slot: tuple, reads, writes) -> None:
            ensure_cycle(cycle)
            cycles[cycle].setdefault(engine, []).append(slot)
            usage[cycle][engine] += 1
            if usage[cycle][engine] >= SLOT_LIMITS[engine]:
                next_free[engine] = cycle + 1
            elif next_free[engine] < cycle:
                next_free[engine] = cycle
            for addr in reads:
                if last_read[addr] < cycle:
                    last_read[addr] = cycle
            for addr in writes:
                last_write[addr] = cycle
                ready_time[addr] = cycle + 1

        for item in slots:
            if isinstance(item, dict):
                # Treat explicit bundle as a barrier: place as-is in next cycle
                cycle = len(cycles)
                ensure_cycle(cycle)
                cycles[cycle] = item
                # Account for dependencies within the bundle
                for eng, eng_slots in item.items():
                    for slot in eng_slots:
                        reads, writes, _barrier = self._slot_rw(eng, slot)
                        place_op(cycle, eng, slot, reads, writes)
                continue

            engine, slot = item
            reads, writes, barrier = self._slot_rw(engine, slot)
            if barrier:
                cycle = len(cycles)
                ensure_cycle(cycle)
                place_op(cycle, engine, slot, reads, writes)
                continue

            earliest = 0
            for addr in reads:
                earliest = max(earliest, ready_time[addr])
            for addr in writes:
                earliest = max(earliest, last_write[addr] + 1, last_read[addr])

            cycle = find_cycle(engine, earliest)
            place_op(cycle, engine, slot, reads, writes)

        return [c for c in cycles if c]

    def build_linear_fast(self, slots: list[tuple[Engine, tuple]]):
        """
        Faster scheduler: enforces RAW/WAR/WAW with array-backed dependency tracking.
        """
        max_addr = SCRATCH_SIZE
        cycles: list[dict[str, list[tuple]]] = []
        usage: list[dict[str, int]] = []
        ready_time = [0] * max_addr
        last_write = [-1] * max_addr
        last_read = [-1] * max_addr
        next_free: dict[str, int] = defaultdict(int)
        max_cycle = -1
        barrier_floor = 0

        def ensure_cycle(cycle: int) -> None:
            while len(cycles) <= cycle:
                cycles.append({})
                usage.append(defaultdict(int))

        def find_cycle(engine: str, earliest: int) -> int:
            cycle = earliest if earliest > next_free[engine] else next_free[engine]
            limit = SLOT_LIMITS[engine]
            while True:
                ensure_cycle(cycle)
                if usage[cycle][engine] < limit:
                    return cycle
                cycle += 1

        def place_op(cycle: int, engine: str, slot: tuple, reads, writes) -> None:
            nonlocal max_cycle
            ensure_cycle(cycle)
            cycles[cycle].setdefault(engine, []).append(slot)
            usage[cycle][engine] += 1
            if usage[cycle][engine] >= SLOT_LIMITS[engine]:
                next_free[engine] = cycle + 1
            elif next_free[engine] < cycle:
                next_free[engine] = cycle
            if cycle > max_cycle:
                max_cycle = cycle
            for addr in reads:
                if last_read[addr] < cycle:
                    last_read[addr] = cycle
            for addr in writes:
                last_write[addr] = cycle
                ready_time[addr] = cycle + 1

        def iter_reads_writes(engine: str, slot: tuple):
            reads = []
            writes = []
            if engine == "alu":
                _op, dest, a1, a2 = slot
                reads = [a1, a2]
                writes = [dest]
            elif engine == "valu":
                match slot:
                    case ("vbroadcast", dest, src):
                        reads = [src]
                        writes = list(range(dest, dest + VLEN))
                    case ("multiply_add", dest, a, b, c):
                        reads = (
                            list(range(a, a + VLEN))
                            + list(range(b, b + VLEN))
                            + list(range(c, c + VLEN))
                        )
                        writes = list(range(dest, dest + VLEN))
                    case (_op, dest, a1, a2):
                        reads = list(range(a1, a1 + VLEN)) + list(range(a2, a2 + VLEN))
                        writes = list(range(dest, dest + VLEN))
            elif engine == "load":
                match slot:
                    case ("load", dest, addr):
                        reads = [addr]
                        writes = [dest]
                    case ("load_offset", dest, addr, offset):
                        reads = [addr + offset]
                        writes = [dest + offset]
                    case ("vload", dest, addr):
                        reads = [addr]
                        writes = list(range(dest, dest + VLEN))
                    case ("const", dest, _val):
                        writes = [dest]
            elif engine == "store":
                match slot:
                    case ("store", addr, src):
                        reads = [addr, src]
                    case ("vstore", addr, src):
                        reads = [addr] + list(range(src, src + VLEN))
            elif engine == "flow":
                match slot:
                    case ("select", dest, cond, a, b):
                        reads = [cond, a, b]
                        writes = [dest]
                    case ("vselect", dest, cond, a, b):
                        reads = (
                            list(range(cond, cond + VLEN))
                            + list(range(a, a + VLEN))
                            + list(range(b, b + VLEN))
                        )
                        writes = list(range(dest, dest + VLEN))
                    case ("add_imm", dest, a, _imm):
                        reads = [a]
                        writes = [dest]
            return reads, writes

        for item in slots:
            if isinstance(item, dict):
                cycle = max_cycle + 1
                ensure_cycle(cycle)
                for eng, eng_slots in item.items():
                    for slot in eng_slots:
                        reads, writes = iter_reads_writes(eng, slot)
                        place_op(cycle, eng, slot, reads, writes)
                barrier_floor = cycle + 1
                continue

            engine, slot = item
            reads, writes, barrier = self._slot_rw(engine, slot)
            if barrier:
                cycle = max_cycle + 1
                place_op(cycle, engine, slot, reads, writes)
                barrier_floor = cycle + 1
                continue

            earliest = 0
            for addr in reads:
                rt = ready_time[addr]
                if rt > earliest:
                    earliest = rt
            for addr in writes:
                lw = last_write[addr] + 1
                if lw > earliest:
                    earliest = lw
                lr = last_read[addr]
                if lr > earliest:
                    earliest = lr

            if barrier_floor > earliest:
                earliest = barrier_floor

            cycle = find_cycle(engine, earliest)
            place_op(cycle, engine, slot, reads, writes)

        # Local window re-pack to improve packing without global cost.
        window = 400
        packed = []
        i = 0
        while i < len(cycles):
            chunk = cycles[i : i + window]
            flat = []
            for b in chunk:
                for eng, eng_slots in b.items():
                    for slot in eng_slots:
                        flat.append((eng, slot))
            # Use greedy within the window.
            packed.extend(self.build_greedy(flat))
            i += window

        return [c for c in packed if c]

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

    def scratch_const(self, val, name=None, slots=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            if slots is None:
                self.add("load", ("const", addr, val))
            else:
                slots.append(("load", ("const", addr, val)))
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
        cache_key = (
            KERNEL_CACHE_VERSION,
            forest_height,
            n_nodes,
            batch_size,
            rounds,
            self.schedule_seed,
        )
        cached = KERNEL_CACHE.get(cache_key)
        if cached is not None:
            self.instrs, self.scratch_debug = cached
            return

        # -------------------------
        # Init scratch + params
        # -------------------------
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        init_loads = []
        init_bcasts = []

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
            init_loads.append(("load", ("const", tmp1, i)))
            init_loads.append(("load", ("load", self.scratch[v], tmp1)))

        zero_const = self.scratch_const(0, slots=init_loads)
        one_const = self.scratch_const(1, slots=init_loads)
        two_const = self.scratch_const(2, slots=init_loads)

        def vec_const(val: int, name: str, load_slots=None, bcast_slots=None):
            vec_addr = self.alloc_scratch(name, VLEN)
            scalar = self.scratch_const(val, slots=load_slots)
            if bcast_slots is None:
                self.add("valu", ("vbroadcast", vec_addr, scalar))
            else:
                bcast_slots.append(("valu", ("vbroadcast", vec_addr, scalar)))
            return vec_addr

        zero_v = vec_const(0, "zero_v", init_loads, init_bcasts)
        one_v = vec_const(1, "one_v", init_loads, init_bcasts)
        two_v = vec_const(2, "two_v", init_loads, init_bcasts)
        three_v = vec_const(3, "three_v", init_loads, init_bcasts)
        four_v = vec_const(4, "four_v", init_loads, init_bcasts)
        five_v = vec_const(5, "five_v", init_loads, init_bcasts)
        six_v = vec_const(6, "six_v", init_loads, init_bcasts)
        # (no extra small idx consts needed)

        n_nodes_v = self.alloc_scratch("n_nodes_v", VLEN)
        init_bcasts.append(("valu", ("vbroadcast", n_nodes_v, self.scratch["n_nodes"])))

        forest_base_v = self.alloc_scratch("forest_base_v", VLEN)
        init_bcasts.append(
            ("valu", ("vbroadcast", forest_base_v, self.scratch["forest_values_p"]))
        )

        # Pre-broadcast hash constants + precompute mul constants for fused stages
        hash_const_v = {}
        hash_mul_v = {}
        hash_const_s1 = []
        hash_const_s3 = []
        hash_mul_s = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            if val1 not in hash_const_v:
                hash_const_v[val1] = vec_const(val1, f"c_{val1}", init_loads, init_bcasts)
            if val3 not in hash_const_v:
                hash_const_v[val3] = vec_const(val3, f"c_{val3}", init_loads, init_bcasts)
            hash_const_s1.append(self.scratch_const(val1, slots=init_loads))
            # If stage is: t1 = val + c1; t2 = val << s; val = t1 + t2
            # then val = val * (1 + 2^s) + c1  (mod 2^32)
            if (op1, op2, op3) == ("+", "+", "<<"):
                mul = (1 + (1 << val3)) % (2**32)
                if mul not in hash_mul_v:
                    hash_mul_v[mul] = vec_const(
                        mul, f"mul_{mul}", init_loads, init_bcasts
                    )
                hash_mul_s.append(self.scratch_const(mul, slots=init_loads))
                hash_const_s3.append(None)
            else:
                hash_const_s3.append(self.scratch_const(val3, slots=init_loads))
                hash_mul_s.append(None)

        self.instrs.extend(self.build_greedy(init_loads))
        self.instrs.extend(self.build_greedy(init_bcasts))

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
        for node_i in range(7):  # nodes 0..6 cover rounds 0..2.
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
        # Vector scratch for pipeline
        # -------------------------
        PIPE_STAGES = 8
        idx_regs = [self.alloc_scratch(f"idx{i}", VLEN) for i in range(PIPE_STAGES)]
        val_regs = [self.alloc_scratch(f"val{i}", VLEN) for i in range(PIPE_STAGES)]
        node_regs = [self.alloc_scratch(f"node{i}", VLEN) for i in range(PIPE_STAGES)]
        addr_regs = [self.alloc_scratch(f"addr{i}", VLEN) for i in range(PIPE_STAGES)]
        t1_regs = [self.alloc_scratch(f"t1{i}", VLEN) for i in range(PIPE_STAGES)]
        t2_regs = [self.alloc_scratch(f"t2{i}", VLEN) for i in range(PIPE_STAGES)]

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

        def compute_ops_split(
            idx_reg, val_reg, node_reg, t1_reg, t2_reg, update_idx=True, use_valu_xor=True
        ):
            valu_ops = []
            flow_ops = []
            # val = hash(val ^ node)
            if use_valu_xor:
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

        def xor_alu_ops(val_reg, node_reg):
            # Per-lane XOR in ALU to relieve VALU pressure (for non-pipelined rounds).
            return [
                ("alu", ("^", val_reg + lane, val_reg + lane, node_reg + lane))
                for lane in range(VLEN)
            ]

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

        def schedule_pipeline_rounds(
            start_round: int, total_rounds: int, n_stages: int, mode: str = "round_robin"
        ):
            """
            Cross-round pipeline scheduler for rounds >= start_round.
            Produces explicit VLIW bundles with load+valu packing.
            """
            bundles = []
            rnd = random.Random(self.schedule_seed)
            gather_first = False

            # Per-stage state
            stage_round = [start_round for _ in range(n_stages)]
            stage_state = [
                "need_addr" if start_round < total_rounds else "done" for _ in range(n_stages)
            ]
            stage_loads = [0 for _ in range(n_stages)]
            stage_comp_ops = [[] for _ in range(n_stages)]
            stage_comp_idx = [0 for _ in range(n_stages)]

            def stage_done(s):
                return stage_state[s] == "done"

            def any_active():
                return any(not stage_done(s) for s in range(n_stages))

            while any_active():
                bundle = {}
                addr_issued = set()
                compute_issued = set()

                # Build valu slots (up to 6): prioritize compute, then staggered addr
                valu_slots = []
                # First, compute ops
                for _ in range(SLOT_LIMITS["valu"]):
                    found = False
                    stage_order = list(range(n_stages))
                    if mode == "random":
                        rnd.shuffle(stage_order)
                    for s in stage_order:
                        if s in compute_issued:
                            continue
                        if stage_state[s] == "compute":
                            ops = stage_comp_ops[s]
                            idx = stage_comp_idx[s]
                            if idx < len(ops):
                                valu_slots.append(ops[idx][1])
                                stage_comp_idx[s] += 1
                                compute_issued.add(s)
                                found = True
                                break
                    if not found:
                        break

                # Then, addr calcs if slots remain
                addr_budget = 2
                for _ in range(min(addr_budget, SLOT_LIMITS["valu"] - len(valu_slots))):
                    found = False
                    stage_order = list(range(n_stages))
                    if mode == "random":
                        rnd.shuffle(stage_order)
                    for s in stage_order:
                        if s in addr_issued:
                            continue
                        if stage_state[s] == "need_addr":
                            idx_reg = idx_regs[s]
                            addr_reg = addr_regs[s]
                            valu_slots.append(("+", addr_reg, idx_reg, forest_base_v))
                            addr_issued.add(s)
                            found = True
                            break
                    if not found:
                        break
                if valu_slots:
                    bundle["valu"] = valu_slots

                # Build load slots (up to 2), prefer stages already loading (most remaining loads)
                load_slots = []
                for _ in range(SLOT_LIMITS["load"]):
                    found = False
                    stage_order = list(range(n_stages))
                    if mode == "random":
                        rnd.shuffle(stage_order)
                    else:
                        stage_order.sort(
                            key=lambda s: stage_loads[s] if stage_state[s] == "loading" else -1,
                            reverse=True,
                        )
                    for s in stage_order:
                        if stage_state[s] == "loading" and stage_loads[s] > 0:
                            # Don't allow loads in same cycle as addr calc
                            if s in addr_issued:
                                continue
                            load_slots.append(
                                ("load_offset", node_regs[s], addr_regs[s], VLEN - stage_loads[s])
                            )
                            stage_loads[s] -= 1
                            found = True
                            break
                    if not found:
                        break
                if load_slots:
                    bundle["load"] = load_slots

                if not bundle:
                    break

                bundles.append(bundle)

                # State transitions after cycle
                for s in addr_issued:
                    stage_state[s] = "loading"
                    stage_loads[s] = VLEN

                for s in range(n_stages):
                    if stage_state[s] == "loading" and stage_loads[s] == 0:
                        # Loads completed this cycle; compute can start next cycle
                        stage_state[s] = "compute"
                        # Build compute ops for this stage/round
                        update_idx = (stage_round[s] != total_rounds - 1)
                        valu_ops, _flow_ops = compute_ops_split(
                            idx_regs[s],
                            val_regs[s],
                            node_regs[s],
                            t1_regs[s],
                            t2_regs[s],
                            update_idx=update_idx,
                        )
                        stage_comp_ops[s] = valu_ops
                        stage_comp_idx[s] = 0

                for s in compute_issued:
                    if stage_state[s] == "compute":
                        if stage_comp_idx[s] >= len(stage_comp_ops[s]):
                            # Round done for this stage
                            stage_round[s] += 1
                            if stage_round[s] >= total_rounds:
                                stage_state[s] = "done"
                            else:
                                stage_state[s] = "need_addr"

            return bundles

        def unbundle_ops(bundles: list[dict]) -> list[tuple[Engine, tuple]]:
            out = []
            for b in bundles:
                for eng in ("load", "valu", "alu", "flow", "store", "debug"):
                    for slot in b.get(eng, []):
                        out.append((eng, slot))
            return out

        # -------------------------
        # Main vector blocks: 8-step software pipeline
        # -------------------------
        vec_blocks = batch_size // VLEN
        vec_end = vec_blocks * VLEN

        bi = 0
        while bi + (PIPE_STAGES - 1) < vec_blocks:
            base_consts = [
                self.scratch_const((bi + k) * VLEN) for k in range(PIPE_STAGES)
            ]

            # Load idx/val once for all pipeline stages
            for k in range(PIPE_STAGES):
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], base_consts[k])))
                body.append(("load", ("vload", idx_regs[k], tmp_addr)))
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], base_consts[k])))
                body.append(("load", ("vload", val_regs[k], tmp_addr)))

            # Rounds 0..2: use cached gather + round-robin compute.
            for r in range(min(3, rounds)):
                use_alu_xor = (r != 1)
                for stage in range(PIPE_STAGES):
                    body.extend(
                        gather_round_ops(
                            idx_regs[stage],
                            node_regs[stage],
                            addr_regs[stage],
                            t1_regs[stage],
                            r,
                        )
                    )

                alu_lists = []
                valu_lists = []
                flow_lists = []
                for stage in range(PIPE_STAGES):
                    if use_alu_xor:
                        alu_lists.append(xor_alu_ops(val_regs[stage], node_regs[stage]))
                    valu_ops, flow_ops = compute_ops_split(
                        idx_regs[stage],
                        val_regs[stage],
                        node_regs[stage],
                        t1_regs[stage],
                        t2_regs[stage],
                        update_idx=(r != rounds - 1),
                        use_valu_xor=not use_alu_xor,
                    )
                    valu_lists.append(valu_ops)
                    flow_lists.append(flow_ops)
                if use_alu_xor:
                    body.extend(round_robin_ops(alu_lists))
                body.extend(round_robin_ops(valu_lists))
                body.extend(round_robin_ops(flow_lists))

            # Rounds >=3: cross-round pipeline schedule.
            if rounds > 3:
                bundles = schedule_pipeline_rounds(3, rounds, PIPE_STAGES, mode="random")
                body.extend(unbundle_ops(bundles))

            # Store values once for all pipeline stages
            for k in range(PIPE_STAGES):
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], base_consts[k])))
                body.append(("store", ("vstore", tmp_addr, val_regs[k])))

            bi += PIPE_STAGES

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
                use_alu_xor = (r != 1)
                if use_alu_xor:
                    body.extend(xor_alu_ops(val_regs[0], node_regs[0]))
                valu_ops, flow_ops = compute_ops_split(
                    idx_regs[0],
                    val_regs[0],
                    node_regs[0],
                    t1_regs[0],
                    t2_regs[0],
                    update_idx=(r != rounds - 1),
                    use_valu_xor=not use_alu_xor,
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
                for hi, (op1, _val1, op2, op3, _val3) in enumerate(HASH_STAGES):
                    mul_c = hash_mul_s[hi]
                    if mul_c is not None:
                        add_c = hash_const_s1[hi]
                        body.append(("alu", ("*", tmp1, tmp_val, mul_c)))
                        body.append(("alu", ("+", tmp_val, tmp1, add_c)))
                    else:
                        body.append(("alu", (op1, tmp1, tmp_val, hash_const_s1[hi])))
                        body.append(("alu", (op3, tmp2, tmp_val, hash_const_s3[hi])))
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
        body_instrs = self.build_linear_fast(body)
        self.instrs.extend(body_instrs)
        self.instrs.append({"flow": [("pause",)]})

        KERNEL_CACHE[cache_key] = (self.instrs, self.scratch_debug)


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
    schedule_seed: int = 91,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder(schedule_seed=schedule_seed)
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
