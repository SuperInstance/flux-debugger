"""
Comprehensive tests for FLUX Debugger.

Covers:
- Breakpoint creation, enabling, disabling, removal
- Conditional breakpoints (eq, ne, gt, lt, gte, lte)
- Register, opcode, PC, cycle breakpoints
- Hit count tracking
- Single-step execution of all opcodes
- Run-to-completion (run_all)
- Run-until-breakpoint (run_until_break)
- Reverse step (undo)
- State inspection (inspect)
- Stack push/pop operations
- Watch expressions
- History tracking
- Edge cases (empty bytecode, unknown opcodes, division by zero)
"""
import pytest
from debugger import (
    FluxDebugger, Breakpoint, BreakpointType, DebugState
)


# ── Helpers ──────────────────────────────────────────────

def make_simple_program():
    """MOVI R0, 10; MOVI R1, 20; ADD R2, R0, R1; HALT"""
    return [0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00]


# ── Breakpoint Tests ────────────────────────────────────

class TestBreakpointCreation:
    def test_pc_breakpoint_defaults(self):
        bp = Breakpoint(bp_type=BreakpointType.PC, value=5)
        assert bp.value == 5
        assert bp.enabled is True
        assert bp.hit_count == 0
        assert bp.condition == "eq"

    def test_register_breakpoint_defaults(self):
        bp = Breakpoint(bp_type=BreakpointType.REGISTER, value=42, register=3)
        assert bp.register == 3
        assert bp.value == 42

    def test_op_breakpoint_defaults(self):
        bp = Breakpoint(bp_type=BreakpointType.OP, value=0x22)
        assert bp.value == 0x22

    def test_cycle_breakpoint_defaults(self):
        bp = Breakpoint(bp_type=BreakpointType.CYCLE, value=1000)
        assert bp.value == 1000


class TestBreakpointShouldBreak:
    def test_pc_breakpoint_hit(self):
        bp = Breakpoint(bp_type=BreakpointType.PC, value=5)
        assert bp.should_break(5, [0]*64, 0, 0) is True

    def test_pc_breakpoint_miss(self):
        bp = Breakpoint(bp_type=BreakpointType.PC, value=5)
        assert bp.should_break(4, [0]*64, 0, 0) is False
        assert bp.should_break(6, [0]*64, 0, 0) is False

    def test_register_breakpoint_eq(self):
        bp = Breakpoint(bp_type=BreakpointType.REGISTER, value=42, register=0, condition="eq")
        assert bp.should_break(0, [42] + [0]*63, 0, 0) is True
        assert bp.should_break(0, [41] + [0]*63, 0, 0) is False

    def test_register_breakpoint_ne(self):
        bp = Breakpoint(bp_type=BreakpointType.REGISTER, value=42, register=0, condition="ne")
        assert bp.should_break(0, [41] + [0]*63, 0, 0) is True
        assert bp.should_break(0, [42] + [0]*63, 0, 0) is False

    def test_register_breakpoint_gt(self):
        bp = Breakpoint(bp_type=BreakpointType.REGISTER, value=10, register=0, condition="gt")
        assert bp.should_break(0, [11]+[0]*63, 0, 0) is True
        assert bp.should_break(0, [10]+[0]*63, 0, 0) is False
        assert bp.should_break(0, [9]+[0]*63, 0, 0) is False

    def test_register_breakpoint_lt(self):
        bp = Breakpoint(bp_type=BreakpointType.REGISTER, value=10, register=0, condition="lt")
        assert bp.should_break(0, [9]+[0]*63, 0, 0) is True
        assert bp.should_break(0, [10]+[0]*63, 0, 0) is False
        assert bp.should_break(0, [11]+[0]*63, 0, 0) is False

    def test_register_breakpoint_gte(self):
        bp = Breakpoint(bp_type=BreakpointType.REGISTER, value=10, register=0, condition="gte")
        assert bp.should_break(0, [10]+[0]*63, 0, 0) is True
        assert bp.should_break(0, [11]+[0]*63, 0, 0) is True
        assert bp.should_break(0, [9]+[0]*63, 0, 0) is False

    def test_register_breakpoint_lte(self):
        bp = Breakpoint(bp_type=BreakpointType.REGISTER, value=10, register=0, condition="lte")
        assert bp.should_break(0, [10]+[0]*63, 0, 0) is True
        assert bp.should_break(0, [9]+[0]*63, 0, 0) is True
        assert bp.should_break(0, [11]+[0]*63, 0, 0) is False

    def test_op_breakpoint_hit(self):
        bp = Breakpoint(bp_type=BreakpointType.OP, value=0x22)  # MUL
        assert bp.should_break(0, [0]*64, 0x22, 0) is True
        assert bp.should_break(0, [0]*64, 0x20, 0) is False

    def test_cycle_breakpoint_before(self):
        bp = Breakpoint(bp_type=BreakpointType.CYCLE, value=100)
        assert bp.should_break(0, [0]*64, 0, 50) is False
        assert bp.should_break(0, [0]*64, 0, 99) is False

    def test_cycle_breakpoint_at(self):
        bp = Breakpoint(bp_type=BreakpointType.CYCLE, value=100)
        assert bp.should_break(0, [0]*64, 0, 100) is True

    def test_cycle_breakpoint_after(self):
        bp = Breakpoint(bp_type=BreakpointType.CYCLE, value=100)
        assert bp.should_break(0, [0]*64, 0, 200) is True

    def test_unknown_condition_defaults_false(self):
        bp = Breakpoint(bp_type=BreakpointType.REGISTER, value=10, register=0, condition="invalid")
        assert bp.should_break(0, [10]+[0]*63, 0, 0) is False


class TestBreakpointEnabled:
    def test_disabled_never_triggers(self):
        bp = Breakpoint(bp_type=BreakpointType.PC, value=5, enabled=False)
        assert bp.should_break(5, [0]*64, 0, 0) is False

    def test_enabled_triggers(self):
        bp = Breakpoint(bp_type=BreakpointType.PC, value=5, enabled=True)
        assert bp.should_break(5, [0]*64, 0, 0) is True


class TestBreakpointHitCount:
    def test_hit_count_increments(self):
        bp = Breakpoint(bp_type=BreakpointType.PC, value=5)
        assert bp.hit_count == 0
        bp.should_break(5, [0]*64, 0, 0)
        assert bp.hit_count == 1
        bp.should_break(5, [0]*64, 0, 0)
        assert bp.hit_count == 2

    def test_disabled_does_not_increment(self):
        bp = Breakpoint(bp_type=BreakpointType.PC, value=5, enabled=False)
        bp.should_break(5, [0]*64, 0, 0)
        assert bp.hit_count == 0

    def test_miss_increments_hit_count(self):
        """should_break increments hit_count regardless of match."""
        bp = Breakpoint(bp_type=BreakpointType.PC, value=5)
        bp.should_break(99, [0]*64, 0, 0)  # miss
        assert bp.hit_count == 1


class TestBreakpointOutOfRangeRegister:
    def test_register_out_of_range(self):
        """Register index beyond register list returns 0."""
        bp = Breakpoint(bp_type=BreakpointType.REGISTER, value=0, register=100, condition="eq")
        assert bp.should_break(0, [0]*64, 0, 0) is True


# ── Debugger Initialization ────────────────────────────

class TestDebuggerInit:
    def test_initial_state(self):
        dbg = FluxDebugger([0x00])
        assert dbg.pc == 0
        assert dbg.cycles == 0
        assert dbg.halted is False
        assert dbg.sp == 4096
        assert len(dbg.registers) == 64
        assert all(r == 0 for r in dbg.registers)

    def test_initial_breakpoints_empty(self):
        dbg = FluxDebugger([0x00])
        assert len(dbg.breakpoints) == 0

    def test_initial_history_empty(self):
        dbg = FluxDebugger([0x00])
        assert len(dbg.history) == 0

    def test_initial_watches_empty(self):
        dbg = FluxDebugger([0x00])
        assert len(dbg.watches) == 0

    def test_empty_bytecode(self):
        dbg = FluxDebugger([])
        assert dbg.pc == 0
        state = dbg.step()
        assert state is None


# ── Single-Step Execution ──────────────────────────────

class TestStepHALT:
    def test_halt_sets_flag(self):
        dbg = FluxDebugger([0x00])
        dbg.step()
        assert dbg.halted is True

    def test_halt_returns_state(self):
        dbg = FluxDebugger([0x00])
        state = dbg.step()
        assert state is not None
        assert state.op_name == "HALT"
        assert state.halted is True

    def test_halt_advances_pc(self):
        dbg = FluxDebugger([0x00])
        dbg.step()
        assert dbg.pc == 1


class TestStepNOP:
    def test_nop_advances_pc(self):
        dbg = FluxDebugger([0x01])
        dbg.step()
        assert dbg.pc == 1
        assert dbg.halted is False

    def test_nop_no_side_effects(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x01, 0x00])
        dbg.step()  # MOVI R0, 42
        dbg.step()  # NOP
        assert dbg.registers[0] == 42


class TestStepMOVI:
    def test_movi_positive(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.step()
        assert dbg.registers[0] == 42

    def test_movi_zero(self):
        dbg = FluxDebugger([0x18, 0, 0, 0x00])
        dbg.step()
        assert dbg.registers[0] == 0

    def test_movi_negative(self):
        """Signed byte: 0xFF = -1."""
        dbg = FluxDebugger([0x18, 0, 0xFF, 0x00])
        dbg.step()
        assert dbg.registers[0] == -1

    def test_movi_max_positive(self):
        dbg = FluxDebugger([0x18, 0, 127, 0x00])
        dbg.step()
        assert dbg.registers[0] == 127

    def test_movi_min_negative(self):
        dbg = FluxDebugger([0x18, 0, 128, 0x00])
        dbg.step()
        assert dbg.registers[0] == -128

    def test_movi_advances_pc_by_3(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.step()
        assert dbg.pc == 3


class TestStepADDI:
    def test_addi_positive(self):
        dbg = FluxDebugger([0x18, 0, 10, 0x19, 0, 5, 0x00])
        dbg.step()  # MOVI R0, 10
        dbg.step()  # ADDI R0, 5
        assert dbg.registers[0] == 15

    def test_addi_negative(self):
        dbg = FluxDebugger([0x18, 0, 10, 0x19, 0, 0xFF, 0x00])  # ADDI R0, -1
        dbg.step()
        dbg.step()
        assert dbg.registers[0] == 9


class TestStepINC:
    def test_inc_from_zero(self):
        dbg = FluxDebugger([0x08, 0, 0x00])
        dbg.step()
        assert dbg.registers[0] == 1

    def test_inc_from_value(self):
        dbg = FluxDebugger([0x18, 0, 99, 0x08, 0, 0x00])
        dbg.step()  # MOVI R0, 99
        dbg.step()  # INC R0
        assert dbg.registers[0] == 100


class TestStepDEC:
    def test_dec_from_zero(self):
        dbg = FluxDebugger([0x09, 0, 0x00])
        dbg.step()
        assert dbg.registers[0] == -1

    def test_dec_from_value(self):
        dbg = FluxDebugger([0x18, 0, 10, 0x09, 0, 0x00])
        dbg.step()  # MOVI R0, 10
        dbg.step()  # DEC R0
        assert dbg.registers[0] == 9


class TestStepNOT:
    def test_not_zero(self):
        dbg = FluxDebugger([0x0A, 0, 0x00])
        dbg.step()
        assert dbg.registers[0] == -1  # ~0 = -1 in Python

    def test_not_value(self):
        dbg = FluxDebugger([0x18, 0, 0x0F, 0x0A, 0, 0x00])
        dbg.step()  # MOVI R0, 15
        dbg.step()  # NOT R0
        assert dbg.registers[0] == ~15


class TestStepNEG:
    def test_neg_positive(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x0B, 0, 0x00])
        dbg.step()  # MOVI R0, 42
        dbg.step()  # NEG R0
        assert dbg.registers[0] == -42

    def test_neg_negative(self):
        dbg = FluxDebugger([0x18, 0, 0xFF, 0x0B, 0, 0x00])  # R0 = -1
        dbg.step()
        dbg.step()
        assert dbg.registers[0] == 1


class TestStepPUSHPOP:
    def test_push_pop_roundtrip(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x0C, 0, 0x0D, 1, 0x00])
        dbg.step()  # MOVI R0, 42
        dbg.step()  # PUSH R0
        dbg.step()  # POP R1
        assert dbg.registers[0] == 42
        assert dbg.registers[1] == 42

    def test_push_decrements_sp(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x0C, 0, 0x00])
        dbg.step()
        dbg.step()
        assert dbg.sp == 4095

    def test_pop_increments_sp(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x0C, 0, 0x0D, 1, 0x00])
        dbg.step(); dbg.step(); dbg.step()
        assert dbg.sp == 4096


class TestStepADD:
    def test_add_simple(self):
        dbg = FluxDebugger([0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00])
        for _ in range(4):
            dbg.step()
        assert dbg.registers[2] == 30

    def test_add_negative(self):
        dbg = FluxDebugger([0x18, 0, 10, 0x18, 1, 0xFF, 0x20, 2, 0, 1, 0x00])  # R1=-1
        for _ in range(4):
            dbg.step()
        assert dbg.registers[2] == 9

    def test_add_with_zero(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x18, 1, 0, 0x20, 2, 0, 1, 0x00])
        for _ in range(4):
            dbg.step()
        assert dbg.registers[2] == 42


class TestStepSUB:
    def test_sub_simple(self):
        dbg = FluxDebugger([0x18, 0, 30, 0x18, 1, 12, 0x21, 2, 0, 1, 0x00])
        for _ in range(4):
            dbg.step()
        assert dbg.registers[2] == 18


class TestStepMUL:
    def test_mul_simple(self):
        dbg = FluxDebugger([0x18, 0, 6, 0x18, 1, 7, 0x22, 2, 0, 1, 0x00])
        for _ in range(4):
            dbg.step()
        assert dbg.registers[2] == 42

    def test_mul_by_zero(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x18, 1, 0, 0x22, 2, 0, 1, 0x00])
        for _ in range(4):
            dbg.step()
        assert dbg.registers[2] == 0


class TestStepDIV:
    def test_div_simple(self):
        dbg = FluxDebugger([0x18, 0, 20, 0x18, 1, 4, 0x23, 2, 0, 1, 0x00])
        for _ in range(4):
            dbg.step()
        assert dbg.registers[2] == 5

    def test_div_truncates(self):
        dbg = FluxDebugger([0x18, 0, 7, 0x18, 1, 2, 0x23, 2, 0, 1, 0x00])
        for _ in range(4):
            dbg.step()
        assert dbg.registers[2] == 3

    def test_div_by_zero(self):
        """Division by zero should not crash."""
        dbg = FluxDebugger([0x18, 0, 42, 0x18, 1, 0, 0x23, 2, 0, 1, 0x00])
        for _ in range(4):
            dbg.step()
        assert dbg.registers[2] == 0  # unchanged from MOVI R2 init


class TestStepCMPGT:
    def test_cmp_gt_true(self):
        dbg = FluxDebugger([0x18, 0, 20, 0x18, 1, 10, 0x2E, 2, 0, 1, 0x00])
        for _ in range(4):
            dbg.step()
        assert dbg.registers[2] == 1

    def test_cmp_gt_false(self):
        dbg = FluxDebugger([0x18, 0, 10, 0x18, 1, 20, 0x2E, 2, 0, 1, 0x00])
        for _ in range(4):
            dbg.step()
        assert dbg.registers[2] == 0

    def test_cmp_gt_equal(self):
        dbg = FluxDebugger([0x18, 0, 10, 0x18, 1, 10, 0x2E, 2, 0, 1, 0x00])
        for _ in range(4):
            dbg.step()
        assert dbg.registers[2] == 0


class TestStepMOV:
    def test_mov(self):
        dbg = FluxDebugger([0x18, 0, 99, 0x3A, 5, 0, 0, 0x00])
        dbg.step()  # MOVI R0, 99
        dbg.step()  # MOV R5, R0, R0
        assert dbg.registers[5] == 99


class TestStepMOVI16:
    def test_mov_i16_positive(self):
        # MOVI16: little-endian imm16. 4096 = 0x1000 -> bytes [0x00, 0x10]
        dbg = FluxDebugger([0x40, 0, 0x00, 0x10, 0x00])  # MOVI16 R0, 4096
        dbg.step()
        assert dbg.registers[0] == 4096

    def test_mov_i16_negative(self):
        dbg = FluxDebugger([0x40, 0, 0x00, 0x80, 0x00])  # MOVI16 R0, -32768
        dbg.step()
        assert dbg.registers[0] == -32768


class TestStepJNZ:
    def test_jnz_taken(self):
        # MOVI R0, 1; JNZ R0, offset(3) ; MOVI R1, 99; HALT
        # JNZ at PC=3, offset=3 -> target=6 (skips MOVI R1)
        dbg = FluxDebugger([0x18, 0, 1, 0x3D, 0, 3, 0, 0x18, 1, 99, 0x00])
        dbg.step()  # MOVI R0, 1
        dbg.step()  # JNZ R0, 3 -> pc = 3 + 3 = 6 (MOVI R1 at offset 6 is skipped)
        assert dbg.pc == 6  # jumped to HALT
        # R1 should still be 0 since MOVI R1 was skipped
        assert dbg.registers[1] == 0

    def test_jnz_not_taken(self):
        # MOVI R0, 0; JNZ R0, offset(6) ; HALT
        dbg = FluxDebugger([0x18, 0, 0, 0x3D, 0, 6, 0, 0x00])
        dbg.step()  # MOVI R0, 0
        dbg.step()  # JNZ R0, 6 -> falls through
        assert dbg.pc == 7


# ── Cycle Counting ─────────────────────────────────────

class TestCycles:
    def test_each_step_increments_cycle(self):
        dbg = FluxDebugger([0x01, 0x01, 0x01, 0x00])  # 3 NOPs, HALT
        assert dbg.cycles == 0
        dbg.step()
        assert dbg.cycles == 1
        dbg.step()
        assert dbg.cycles == 2
        dbg.step()
        assert dbg.cycles == 3

    def test_cycles_in_state(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        state = dbg.step()
        assert state.cycles == 1


# ── Run All ────────────────────────────────────────────

class TestRunAll:
    def test_run_all_completes(self):
        dbg = FluxDebugger(make_simple_program())
        states = dbg.run_all()
        assert dbg.halted is True
        assert dbg.registers[2] == 30

    def test_run_all_returns_states(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        states = dbg.run_all()
        assert len(states) == 2  # MOVI + HALT

    def test_run_all_empty_program(self):
        dbg = FluxDebugger([])
        states = dbg.run_all()
        assert len(states) == 0


# ── Run Until Break ────────────────────────────────────

class TestRunUntilBreak:
    def test_pc_breakpoint_stops(self):
        dbg = FluxDebugger(make_simple_program())
        dbg.add_breakpoint(Breakpoint(bp_type=BreakpointType.PC, value=6))
        states = dbg.run_until_break()
        assert dbg.pc == 6

    def test_register_breakpoint_stops(self):
        dbg = FluxDebugger(make_simple_program())
        dbg.add_breakpoint(Breakpoint(bp_type=BreakpointType.REGISTER, value=30, register=2))
        states = dbg.run_until_break()
        assert dbg.registers[2] == 30

    def test_op_breakpoint_stops(self):
        dbg = FluxDebugger(make_simple_program())
        dbg.add_breakpoint(Breakpoint(bp_type=BreakpointType.OP, value=0x20))  # ADD
        states = dbg.run_until_break()
        assert len(states) > 0

    def test_cycle_breakpoint_stops(self):
        dbg = FluxDebugger([0x01, 0x01, 0x01, 0x01, 0x01, 0x00])
        dbg.add_breakpoint(Breakpoint(bp_type=BreakpointType.CYCLE, value=3))
        states = dbg.run_until_break()
        assert dbg.cycles >= 3

    def test_no_breakpoints_runs_all(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        states = dbg.run_until_break()
        assert dbg.halted is True


# ── Breakpoint Management ──────────────────────────────

class TestBreakpointManagement:
    def test_add_breakpoint(self):
        dbg = FluxDebugger([0x00])
        bp = Breakpoint(bp_type=BreakpointType.PC, value=0)
        dbg.add_breakpoint(bp)
        assert len(dbg.breakpoints) == 1

    def test_remove_breakpoint(self):
        dbg = FluxDebugger([0x00])
        dbg.add_breakpoint(Breakpoint(bp_type=BreakpointType.PC, value=0))
        dbg.add_breakpoint(Breakpoint(bp_type=BreakpointType.PC, value=1))
        dbg.remove_breakpoint(0)
        assert len(dbg.breakpoints) == 1

    def test_remove_breakpoint_invalid_index(self):
        dbg = FluxDebugger([0x00])
        dbg.remove_breakpoint(0)  # no crash
        assert len(dbg.breakpoints) == 0

    def test_remove_breakpoint_negative_index(self):
        dbg = FluxDebugger([0x00])
        dbg.add_breakpoint(Breakpoint(bp_type=BreakpointType.PC, value=0))
        dbg.remove_breakpoint(-1)
        assert len(dbg.breakpoints) == 1  # -1 is not >= 0

    def test_multiple_breakpoints(self):
        dbg = FluxDebugger([0x00])
        dbg.add_breakpoint(Breakpoint(bp_type=BreakpointType.PC, value=0))
        dbg.add_breakpoint(Breakpoint(bp_type=BreakpointType.PC, value=1))
        dbg.add_breakpoint(Breakpoint(bp_type=BreakpointType.CYCLE, value=10))
        assert len(dbg.breakpoints) == 3


# ── Watch Expressions ─────────────────────────────────

class TestWatchExpressions:
    def test_add_watch(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.add_watch("R0", lambda d: d.registers[0])
        assert "R0" in dbg.watches

    def test_watch_callable(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.add_watch("R0", lambda d: d.registers[0])
        dbg.step()
        assert dbg.watches["R0"](dbg) == 42


# ── Reverse Step ───────────────────────────────────────

class TestReverseStep:
    def test_reverse_single(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.step()  # MOVI
        dbg.step()  # HALT
        prev = dbg.reverse()
        assert prev is not None
        assert dbg.halted is False

    def test_reverse_no_history(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        prev = dbg.reverse()
        assert prev is None

    def test_reverse_only_one_step(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.step()
        prev = dbg.reverse()
        assert prev is None  # need at least 2 entries

    def test_reverse_restores_registers(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x18, 1, 99, 0x00])
        dbg.step()  # MOVI R0, 42
        dbg.step()  # MOVI R1, 99
        dbg.step()  # HALT
        dbg.reverse()
        assert dbg.registers[1] == 99
        dbg.reverse()
        assert dbg.registers[1] == 0  # restored to initial

    def test_reverse_restores_pc(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.step()  # pc=3
        dbg.step()  # pc=4
        dbg.reverse()
        assert dbg.pc == 3


# ── Inspect ────────────────────────────────────────────

class TestInspect:
    def test_inspect_initial(self):
        dbg = FluxDebugger([0x00])
        info = dbg.inspect()
        assert info["pc"] == 0
        assert info["cycles"] == 0
        assert info["halted"] is False
        assert info["registers"]["R0"] == 0
        assert info["sp"] == 4096
        assert info["breakpoints"] == 0
        assert info["history_size"] == 0

    def test_inspect_after_step(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.step()
        info = dbg.inspect()
        assert info["registers"]["R0"] == 42
        assert info["cycles"] == 1

    def test_inspect_has_all_eight_registers(self):
        dbg = FluxDebugger([0x00])
        info = dbg.inspect()
        for i in range(8):
            assert f"R{i}" in info["registers"]


# ── Step Over ──────────────────────────────────────────

class TestStepOver:
    def test_step_over_equals_step(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.step()  # MOVI
        state = dbg.step_over()  # HALT
        assert state is not None
        assert dbg.halted is True


# ── History ────────────────────────────────────────────

class TestHistory:
    def test_history_grows_with_steps(self):
        dbg = FluxDebugger([0x01, 0x01, 0x00])
        dbg.step()
        dbg.step()
        assert len(dbg.history) == 2

    def test_history_states_have_correct_fields(self):
        # NOP (0x01) is a 1-byte instruction, so _save_state reads
        # bytecode[pc-1] which IS the opcode byte.
        dbg = FluxDebugger([0x01, 0x00])
        state = dbg.step()
        assert isinstance(state, DebugState)
        assert state.op_name == "NOP"
        assert isinstance(state.registers, list)


# ── Integration: Multi-Breakpoint ──────────────────────

class TestIntegrationMultiBreakpoint:
    def test_multiple_breakpoints_first_wins(self):
        """With two PC breakpoints, run_until_break stops at the first hit."""
        dbg = FluxDebugger([0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00])
        dbg.add_breakpoint(Breakpoint(bp_type=BreakpointType.PC, value=3))
        dbg.add_breakpoint(Breakpoint(bp_type=BreakpointType.PC, value=6))
        states = dbg.run_until_break()
        assert dbg.pc == 3


# ── Integration: Factorial with Breakpoints ────────────

class TestIntegrationFactorial:
    def test_factorial_mul_breakpoint(self):
        bc = [0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0,
              0x09, 0, 0x3D, 0, 0xFA, 0, 0x00]
        dbg = FluxDebugger(bc)
        dbg.add_breakpoint(Breakpoint(bp_type=BreakpointType.OP, value=0x22))  # MUL
        states = dbg.run_until_break()
        assert len(states) > 0

    def test_factorial_cycle_breakpoint(self):
        bc = [0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0,
              0x09, 0, 0x3D, 0, 0xFA, 0, 0x00]
        dbg = FluxDebugger(bc)
        dbg.add_breakpoint(Breakpoint(bp_type=BreakpointType.CYCLE, value=10))
        states = dbg.run_until_break()
        assert dbg.cycles >= 10


# ── OP_NAMES coverage ─────────────────────────────────

class TestOpNames:
    def test_known_ops_have_names(self):
        dbg = FluxDebugger([0x00])
        known = {0x00, 0x01, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                 0x18, 0x19, 0x1A, 0x20, 0x21, 0x22, 0x23, 0x24,
                 0x25, 0x26, 0x27, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E,
                 0x2F, 0x3A, 0x3C, 0x3D, 0x40, 0x43, 0x46}
        for op in known:
            assert op in FluxDebugger.OP_NAMES, f"Missing name for opcode {op:#x}"


# ── DebugState ─────────────────────────────────────────

class TestDebugState:
    def test_debug_state_fields(self):
        state = DebugState(
            pc=0, op=0x00, op_name="HALT",
            registers=[0]*16, stack=[], sp=4096,
            cycles=0, halted=False
        )
        assert state.pc == 0
        assert state.op_name == "HALT"
        assert state.note == ""  # default

    def test_debug_state_with_note(self):
        state = DebugState(
            pc=0, op=0x01, op_name="NOP",
            registers=[0]*16, stack=[], sp=4096,
            cycles=1, halted=False, note="test note"
        )
        assert state.note == "test note"
