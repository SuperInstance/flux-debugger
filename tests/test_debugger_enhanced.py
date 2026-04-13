"""Comprehensive pytest tests for enhanced flux-debugger."""

import pytest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from debugger import (
    FluxDebugger, Breakpoint, BreakpointType, DebugState, WatchResult, TraceEntry,
)


# ── Breakpoint management (set/remove/disable by PC or label) ──────────────

class TestBreakpointManagement:
    def test_add_pc_breakpoint(self):
        dbg = FluxDebugger([0x00])
        bp_id = dbg.add_pc_breakpoint(0)
        assert bp_id > 0
        assert len(dbg.breakpoints) == 1

    def test_add_label_breakpoint(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.add_label("init", 0)
        bp_id = dbg.add_label_breakpoint("init")
        assert bp_id > 0
        bp = dbg.get_breakpoint(bp_id)
        assert bp is not None
        assert bp.label == "init"

    def test_add_opcode_breakpoint(self):
        dbg = FluxDebugger([0x00])
        bp_id = dbg.add_opcode_breakpoint(0x22)  # MUL
        assert bp_id > 0
        bp = dbg.get_breakpoint(bp_id)
        assert bp.bp_type == BreakpointType.OP

    def test_add_cycle_breakpoint(self):
        dbg = FluxDebugger([0x00])
        bp_id = dbg.add_cycle_breakpoint(100)
        bp = dbg.get_breakpoint(bp_id)
        assert bp.bp_type == BreakpointType.CYCLE
        assert bp.value == 100

    def test_remove_by_id(self):
        dbg = FluxDebugger([0x00])
        bp_id = dbg.add_pc_breakpoint(0)
        result = dbg.remove_breakpoint(bp_id)
        assert result is True
        assert len(dbg.breakpoints) == 0

    def test_remove_nonexistent(self):
        dbg = FluxDebugger([0x00])
        result = dbg.remove_breakpoint(999)
        assert result is False

    def test_disable_breakpoint(self):
        dbg = FluxDebugger([0x00])
        bp_id = dbg.add_pc_breakpoint(0)
        result = dbg.disable_breakpoint(bp_id)
        assert result is True
        assert dbg.get_breakpoint(bp_id).enabled is False

    def test_enable_breakpoint(self):
        dbg = FluxDebugger([0x00])
        bp_id = dbg.add_pc_breakpoint(0)
        dbg.disable_breakpoint(bp_id)
        dbg.enable_breakpoint(bp_id)
        assert dbg.get_breakpoint(bp_id).enabled is True

    def test_disable_nonexistent(self):
        dbg = FluxDebugger([0x00])
        result = dbg.disable_breakpoint(999)
        assert result is False

    def test_clear_breakpoints(self):
        dbg = FluxDebugger([0x00])
        dbg.add_pc_breakpoint(0)
        dbg.add_pc_breakpoint(3)
        dbg.clear_breakpoints()
        assert len(dbg.breakpoints) == 0

    def test_list_breakpoints(self):
        dbg = FluxDebugger([0x00])
        dbg.add_pc_breakpoint(0)
        dbg.add_label_breakpoint("test")
        dbg.add_label("test", 5)
        bps = dbg.list_breakpoints()
        assert len(bps) == 2
        for bp in bps:
            assert "id" in bp
            assert "type" in bp
            assert "enabled" in bp
            assert "hit_count" in bp

    def test_breakpoint_ids_increment(self):
        dbg = FluxDebugger([0x00])
        id1 = dbg.add_pc_breakpoint(0)
        id2 = dbg.add_pc_breakpoint(3)
        assert id2 > id1

    def test_conditional_breakpoint_on_register(self):
        dbg = FluxDebugger([0x00])
        bp_id = dbg.add_conditional_breakpoint(register=0, value=42, condition="eq")
        bp = dbg.get_breakpoint(bp_id)
        assert bp.bp_type == BreakpointType.REGISTER
        assert bp.value == 42
        assert bp.condition == "eq"

    def test_get_breakpoint_none(self):
        dbg = FluxDebugger([0x00])
        assert dbg.get_breakpoint(999) is None


# ── Step execution (step-in, step-over, step-out) ─────────────────────────

class TestStepExecution:
    def test_step_in(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        state = dbg.step()
        assert state is not None
        assert state.op_name == "MOVI"

    def test_step_over_normal(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x18, 1, 20, 0x00])
        state = dbg.step_over()
        assert state is not None
        assert state.op_name == "MOVI"

    def test_step_over_skips_call_body(self):
        """step_over on CALL should execute past the call until return."""
        bc = bytearray()
        bc.extend([0x18, 0, 1])   # PC 0: MOVI R0, 1
        bc.append(0x48)           # PC 3: CALL
        bc.append(7)              # target PC 7
        bc.append(0x00)           # PC 5: HALT (after return)
        bc.append(0x01)           # PC 6: NOP padding
        bc.append(0x49)           # PC 7: RET (callee)
        bc.append(0x00)           # PC 8: safety HALT
        dbg = FluxDebugger(list(bc))
        dbg.step()  # MOVI (advance to CALL)
        assert dbg.pc == 3
        state = dbg.step_over()   # Should step over the CALL
        # After step_over, we should be past the CALL (at HALT at PC 5)
        assert dbg.pc == 5

    def test_step_out_from_function(self):
        bc = bytearray()
        bc.extend([0x18, 0, 1])   # PC 0: MOVI R0, 1
        bc.append(0x48)           # PC 3: CALL
        bc.append(10)             # target PC 10
        bc.append(0x00)           # PC 5: HALT
        while len(bc) < 10:
            bc.append(0x01)       # NOP padding
        bc.append(0x49)           # PC 10: RET
        bc.append(0x00)           # PC 11: HALT (shouldn't reach)
        dbg = FluxDebugger(list(bc))
        dbg.step()                # MOVI
        dbg.step()                # CALL (enters function)
        assert dbg.call_depth == 1
        state = dbg.step_out()    # Should return from function
        assert dbg.call_depth == 0

    def test_step_out_at_top_level(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        state = dbg.step_out()
        assert state is not None

    def test_multiple_steps(self):
        dbg = FluxDebugger([0x01, 0x01, 0x01, 0x00])
        count = 0
        while True:
            state = dbg.step()
            if state is None:
                break
            count += 1
        assert count == 4

    def test_step_returns_none_when_halted(self):
        dbg = FluxDebugger([0x00])
        dbg.step()  # HALT
        state = dbg.step()
        assert state is None


# ── Watch expressions ──────────────────────────────────────────────────────

class TestWatchExpressions:
    def test_register_watch(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.add_register_watch("r0_val", 0)
        results = dbg.evaluate_watches()
        assert len(results) == 1
        assert results[0].value == 0

    def test_register_watch_after_change(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.add_register_watch("r0_val", 0)
        dbg.step()  # MOVI R0, 42
        results = dbg.evaluate_watches()
        assert results[0].value == 42

    def test_watch_change_detection(self):
        dbg = FluxDebugger([0x18, 0, 10, 0x19, 0, 5, 0x00])
        dbg.add_register_watch("r0_val", 0)
        dbg.evaluate_watches()  # initial: 0
        dbg.step()              # MOVI R0, 10
        results = dbg.evaluate_watches()
        assert results[0].changed is True
        assert results[0].value == 10

    def test_watch_no_change(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x18, 1, 20, 0x00])
        dbg.add_register_watch("r2_val", 2)
        dbg.evaluate_watches()  # initial
        dbg.step()              # MOVI R0
        dbg.step()              # MOVI R1
        results = dbg.evaluate_watches()
        assert results[0].changed is False

    def test_memory_watch(self):
        dbg = FluxDebugger([0x00])
        dbg.write_memory(100, [42])
        dbg.add_memory_watch("mem_100", 100)
        results = dbg.evaluate_watches()
        assert results[0].value == 42

    def test_custom_watch(self):
        dbg = FluxDebugger([0x18, 0, 10, 0x18, 1, 20, 0x00])
        dbg.add_watch("sum", lambda d: d.registers[0] + d.registers[1])
        dbg.step()
        dbg.step()
        results = dbg.evaluate_watches()
        assert results[0].value == 30

    def test_remove_watch(self):
        dbg = FluxDebugger([0x00])
        dbg.add_register_watch("r0", 0)
        dbg.remove_watch("r0")
        results = dbg.evaluate_watches()
        assert len(results) == 0

    def test_watch_result_type(self):
        dbg = FluxDebugger([0x00])
        dbg.add_register_watch("reg_r0", 0)
        results = dbg.evaluate_watches()
        assert isinstance(results[0], WatchResult)
        assert results[0].name == "reg_r0"
        assert results[0].watch_type == "register"


# ── Stack inspection ───────────────────────────────────────────────────────

class TestStackInspection:
    def test_empty_stack(self):
        dbg = FluxDebugger([0x00])
        stack = dbg.get_stack()
        assert stack == []

    def test_stack_after_push(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x0C, 0, 0x00])
        dbg.step()  # MOVI
        dbg.step()  # PUSH
        stack = dbg.get_stack()
        assert len(stack) == 1
        assert stack[0] == (0, 42)

    def test_stack_depth(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x0C, 0, 0x00])
        assert dbg.get_stack_depth() == 0
        dbg.step()
        dbg.step()
        assert dbg.get_stack_depth() == 1

    def test_stack_multiple_pushes(self):
        dbg = FluxDebugger([0x18, 0, 10, 0x18, 1, 20, 0x0C, 0, 0x0C, 1, 0x00])
        dbg.step()  # MOVI R0, 10
        dbg.step()  # MOVI R1, 20
        dbg.step()  # PUSH R0
        dbg.step()  # PUSH R1
        stack = dbg.get_stack()
        assert len(stack) == 2
        # stack[0] is the top of stack (most recently pushed), stack[1] is bottom
        assert stack[0][1] == 20  # top (last pushed)
        assert stack[1][1] == 10  # bottom (first pushed)

    def test_stack_after_pop(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x0C, 0, 0x0D, 1, 0x00])
        dbg.step()  # MOVI
        dbg.step()  # PUSH
        dbg.step()  # POP
        assert dbg.get_stack_depth() == 0

    def test_call_stack(self):
        bc = bytearray()
        bc.extend([0x18, 0, 1])
        bc.append(0x48)   # CALL 10
        bc.append(10)
        bc.append(0x00)   # HALT
        while len(bc) < 10:
            bc.append(0x01)
        bc.append(0x49)   # RET
        dbg = FluxDebugger(list(bc))
        dbg.step()  # MOVI
        dbg.step()  # CALL
        assert len(dbg.get_call_stack()) == 1


# ── Memory inspection and modification ─────────────────────────────────────

class TestMemoryInspection:
    def test_read_memory_default_zero(self):
        dbg = FluxDebugger([0x00])
        assert dbg.read_memory(100) == [0]

    def test_write_and_read_memory(self):
        dbg = FluxDebugger([0x00])
        dbg.write_memory(50, [0xDE, 0xAD])
        assert dbg.read_memory(50, 2) == [0xDE, 0xAD]

    def test_read_memory_word(self):
        dbg = FluxDebugger([0x00])
        dbg.write_memory_word(100, 0x1234)
        assert dbg.read_memory_word(100) == 0x1234

    def test_write_memory_word(self):
        dbg = FluxDebugger([0x00])
        dbg.write_memory_word(200, 0xBEEF)
        assert dbg.memory[200] == 0xEF
        assert dbg.memory[201] == 0xBE

    def test_memory_dump(self):
        dbg = FluxDebugger([0x00])
        dbg.write_memory(10, [1, 2, 3, 4, 5])
        dump = dbg.memory_dump(10, 15)
        assert len(dump) == 5
        assert dump[0] == (10, 1)
        assert dump[4] == (14, 5)

    def test_out_of_bounds_read(self):
        dbg = FluxDebugger([0x00])
        result = dbg.read_memory(-1)
        assert result == [0]

    def test_out_of_bounds_write(self):
        dbg = FluxDebugger([0x00])
        dbg.write_memory(99999, [42])  # should not crash
        # memory only goes to 65536
        assert dbg.read_memory(99999) == [0]


# ── Execution trace recording ──────────────────────────────────────────────

class TestExecutionTrace:
    def test_trace_disabled_by_default(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.run_all()
        assert len(dbg.get_trace()) == 0

    def test_trace_enabled(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.enable_trace()
        dbg.run_all()
        trace = dbg.get_trace()
        assert len(trace) == 2  # MOVI + HALT

    def test_trace_entries_have_fields(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.enable_trace()
        dbg.run_all()
        trace = dbg.get_trace()
        for entry in trace:
            assert "cycle" in entry
            assert "pc" in entry
            assert "op" in entry
            assert "op_name" in entry

    def test_trace_max_entries(self):
        dbg = FluxDebugger([0x01] * 200 + [0x00])
        dbg.enable_trace(max_entries=10)
        dbg.run_all()
        trace = dbg.get_trace()
        assert len(trace) == 10

    def test_trace_clear(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.enable_trace()
        dbg.run_all()
        assert len(dbg.get_trace()) > 0
        dbg.clear_trace()
        assert len(dbg.get_trace()) == 0

    def test_trace_json(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.enable_trace()
        dbg.run_all()
        j = dbg.get_trace_json()
        data = json.loads(j)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_trace_registers_snapshot(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.enable_trace()
        dbg.step()  # MOVI R0, 42
        trace = dbg.get_trace()
        assert trace[0]["registers_snapshot"]["R0"] == 0  # before execution
        dbg.step()  # HALT
        trace = dbg.get_trace()
        assert trace[1]["registers_snapshot"]["R0"] == 42


# ── Labels ─────────────────────────────────────────────────────────────────

class TestLabels:
    def test_add_and_resolve_label(self):
        dbg = FluxDebugger([0x00])
        dbg.add_label("main", 0)
        assert dbg.resolve_label("main") == 0

    def test_resolve_nonexistent_label(self):
        dbg = FluxDebugger([0x00])
        assert dbg.resolve_label("missing") is None

    def test_label_in_inspect(self):
        dbg = FluxDebugger([0x00])
        dbg.add_label("entry", 0)
        info = dbg.inspect()
        assert "entry" in info["labels"]

    def test_label_breakpoint_triggers(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.add_label("halt_pc", 3)
        dbg.add_label_breakpoint("halt_pc")
        states = dbg.run_until_break()
        assert dbg.pc == 3


# ── Disassembly ────────────────────────────────────────────────────────────

class TestDisassembly:
    def test_disassemble_at(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        d = dbg.disassemble_at(0)
        assert d is not None
        assert d["name"] == "MOVI"
        assert d["operands"] == "R0, 42"

    def test_disassemble_at_halt(self):
        dbg = FluxDebugger([0x00])
        d = dbg.disassemble_at(0)
        assert d["name"] == "HALT"

    def test_disassemble_out_of_range(self):
        dbg = FluxDebugger([0x00])
        assert dbg.disassemble_at(100) is None

    def test_disassemble_with_label(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.add_label("init", 0)
        d = dbg.disassemble_at(0)
        assert d["label"] == "init"

    def test_disassemble_call_with_label(self):
        bc = bytearray()
        bc.extend([0x18, 0, 1])
        bc.append(0x48)   # CALL
        bc.append(10)     # target
        bc.append(0x00)
        while len(bc) < 10:
            bc.append(0x01)
        bc.append(0x00)
        dbg = FluxDebugger(list(bc))
        dbg.add_label("func", 10)
        d = dbg.disassemble_at(3)
        assert "func" in d["operands"]

    def test_disassemble_range(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x18, 1, 20, 0x00])
        instructions = dbg.disassemble_range(0, 10)
        assert len(instructions) == 3  # MOVI, MOVI, HALT
        assert instructions[0]["name"] == "MOVI"
        assert instructions[2]["name"] == "HALT"


# ── CALL/RET with call depth ───────────────────────────────────────────────

class TestCallRet:
    def test_call_depth_increments(self):
        bc = bytearray()
        bc.extend([0x18, 0, 1])
        bc.append(0x48)
        bc.append(10)
        bc.append(0x00)
        while len(bc) < 10:
            bc.append(0x01)
        bc.append(0x49)  # RET
        bc.append(0x00)
        dbg = FluxDebugger(list(bc))
        dbg.step()  # MOVI
        assert dbg.call_depth == 0
        dbg.step()  # CALL
        assert dbg.call_depth == 1
        dbg.step()  # RET
        assert dbg.call_depth == 0

    def test_nested_calls(self):
        bc = bytearray()
        bc.extend([0x18, 0, 1])  # PC 0: MOVI R0, 1
        bc.append(0x48)           # PC 3: CALL 7
        bc.append(7)              # target PC 7
        bc.append(0x00)           # PC 5: HALT
        bc.append(0x01)           # PC 6: NOP
        # Outer callee at PC 7
        bc.append(0x48)           # PC 7: CALL 10
        bc.append(10)             # target PC 10
        bc.append(0x49)           # PC 9: RET (outer callee)
        # Inner callee at PC 10
        bc.append(0x49)           # PC 10: RET (inner callee)
        bc.append(0x00)           # PC 11: safety HALT
        dbg = FluxDebugger(list(bc))
        dbg.step()  # MOVI → PC 3
        assert dbg.call_depth == 0
        dbg.step()  # CALL 7 → PC 7
        assert dbg.call_depth == 1
        assert dbg.pc == 7
        dbg.step()  # CALL 10 → PC 10
        assert dbg.call_depth == 2
        assert dbg.pc == 10
        dbg.step()  # RET → PC 9
        assert dbg.call_depth == 1
        assert dbg.pc == 9
        dbg.step()  # RET → PC 5
        assert dbg.call_depth == 0
        assert dbg.pc == 5

    def test_call_state_has_depth(self):
        bc = bytearray()
        bc.extend([0x18, 0, 1])
        bc.append(0x48)
        bc.append(10)
        bc.append(0x00)
        while len(bc) < 10:
            bc.append(0x01)
        bc.append(0x49)
        bc.append(0x00)
        dbg = FluxDebugger(list(bc))
        dbg.step()  # MOVI
        dbg.step()  # CALL
        state = dbg._save_state(0x01)  # save state inside call
        assert state.call_depth == 1


# ── Additional opcodes ─────────────────────────────────────────────────────

class TestAdditionalOpcodes:
    def test_mod(self):
        dbg = FluxDebugger([0x18, 0, 10, 0x18, 1, 3, 0x24, 2, 0, 1, 0x00])
        dbg.step(); dbg.step(); dbg.step(); dbg.step()  # MOVI x2, MOD
        assert dbg.registers[2] == 1  # 10 % 3

    def test_and(self):
        dbg = FluxDebugger([0x18, 0, 0xFF, 0x18, 1, 0x0F, 0x25, 2, 0, 1, 0x00])
        dbg.step(); dbg.step(); dbg.step(); dbg.step()
        assert dbg.registers[2] == 0x0F

    def test_or(self):
        # Use small values to avoid signed-byte interpretation issues
        dbg = FluxDebugger([0x18, 0, 10, 0x18, 1, 3, 0x26, 2, 0, 1, 0x00])
        dbg.step(); dbg.step(); dbg.step(); dbg.step()
        assert dbg.registers[2] == 11  # 10 | 3

    def test_xor(self):
        dbg = FluxDebugger([0x18, 0, 0xFF, 0x18, 1, 0xFF, 0x27, 2, 0, 1, 0x00])
        dbg.step(); dbg.step(); dbg.step(); dbg.step()
        assert dbg.registers[2] == 0

    def test_cmp_eq(self):
        dbg = FluxDebugger([0x18, 0, 5, 0x18, 1, 5, 0x2C, 2, 0, 1, 0x00])
        dbg.step(); dbg.step(); dbg.step(); dbg.step()
        assert dbg.registers[2] == 1  # equal

    def test_cmp_ne(self):
        dbg = FluxDebugger([0x18, 0, 5, 0x18, 1, 3, 0x2F, 2, 0, 1, 0x00])
        dbg.step(); dbg.step(); dbg.step(); dbg.step()
        assert dbg.registers[2] == 1  # not equal

    def test_not(self):
        dbg = FluxDebugger([0x18, 0, 0, 0x0A, 0, 0x00])
        dbg.step(); dbg.step()
        assert dbg.registers[0] == -1  # ~0

    def test_neg(self):
        dbg = FluxDebugger([0x18, 0, 5, 0x0B, 0, 0x00])
        dbg.step(); dbg.step()
        assert dbg.registers[0] == -5
