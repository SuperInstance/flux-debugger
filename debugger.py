"""
FLUX Step Debugger — debug FLUX bytecodes across all runtimes.

Features:
- Step through bytecodes one instruction at a time
- Set breakpoints on PC values
- Inspect registers, stack, memory at any point
- Conditional breakpoints (register value checks)
- Watch expressions
- Reverse step (undo last instruction)
"""
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple, Callable
from enum import Enum, auto


class BreakpointType(Enum):
    PC = auto()       # Break at specific PC
    REGISTER = auto() # Break when register matches condition
    OP = auto()       # Break on specific opcode
    CYCLE = auto()    # Break after N cycles


@dataclass
class Breakpoint:
    bp_type: BreakpointType
    value: int
    register: int = 0
    condition: str = "eq"  # eq, ne, gt, lt, gte, lte
    enabled: bool = True
    hit_count: int = 0
    
    def should_break(self, pc: int, registers: list, op: int, cycles: int) -> bool:
        if not self.enabled:
            return False
        self.hit_count += 1
        
        if self.bp_type == BreakpointType.PC:
            return pc == self.value
        elif self.bp_type == BreakpointType.REGISTER:
            val = registers[self.register] if self.register < len(registers) else 0
            return self._check_condition(val)
        elif self.bp_type == BreakpointType.OP:
            return op == self.value
        elif self.bp_type == BreakpointType.CYCLE:
            return cycles >= self.value
        return False
    
    def _check_condition(self, val: int) -> bool:
        ops = {
            "eq": val == self.value,
            "ne": val != self.value,
            "gt": val > self.value,
            "lt": val < self.value,
            "gte": val >= self.value,
            "lte": val <= self.value,
        }
        return ops.get(self.condition, False)


@dataclass
class DebugState:
    """Snapshot of VM state at a single point."""
    pc: int
    op: int
    op_name: str
    registers: List[int]
    stack: List[int]
    sp: int
    cycles: int
    halted: bool
    note: str = ""


class FluxDebugger:
    """Step debugger for FLUX bytecodes."""
    
    OP_NAMES = {
        0x00: "HALT", 0x01: "NOP", 0x08: "INC", 0x09: "DEC",
        0x0A: "NOT", 0x0B: "NEG", 0x0C: "PUSH", 0x0D: "POP",
        0x17: "STRIPCONF", 0x18: "MOVI", 0x19: "ADDI", 0x1A: "SUBI",
        0x20: "ADD", 0x21: "SUB", 0x22: "MUL", 0x23: "DIV", 0x24: "MOD",
        0x25: "AND", 0x26: "OR", 0x27: "XOR",
        0x2A: "MIN", 0x2B: "MAX",
        0x2C: "CMP_EQ", 0x2D: "CMP_LT", 0x2E: "CMP_GT", 0x2F: "CMP_NE",
        0x3A: "MOV", 0x3C: "JZ", 0x3D: "JNZ",
        0x40: "MOVI16", 0x43: "JMP", 0x46: "LOOP",
    }
    
    def __init__(self, bytecode: list):
        self.bytecode = bytes(bytecode)
        self.registers = [0] * 64
        self.stack = [0] * 4096
        self.sp = 4096
        self.pc = 0
        self.cycles = 0
        self.halted = False
        self.breakpoints: List[Breakpoint] = []
        self.history: List[DebugState] = []
        self.watches: Dict[str, Callable] = {}
    
    def add_breakpoint(self, bp: Breakpoint):
        self.breakpoints.append(bp)
    
    def remove_breakpoint(self, index: int):
        if 0 <= index < len(self.breakpoints):
            self.breakpoints.pop(index)
    
    def add_watch(self, name: str, fn: Callable):
        self.watches[name] = fn
    
    def _save_state(self, op: int, note: str = "") -> DebugState:
        return DebugState(
            pc=self.pc,
            op=op,
            op_name=self.OP_NAMES.get(op, f"UNKNOWN({op:#x})"),
            registers=self.registers[:16].copy(),
            stack=self.stack[max(self.sp, 0):self.sp+8].copy() if self.sp < 4096 else [],
            sp=self.sp,
            cycles=self.cycles,
            halted=self.halted,
            note=note,
        )
    
    def _check_breakpoints(self) -> bool:
        hit = False
        for bp in self.breakpoints:
            if bp.should_break(self.pc, self.registers, 
                             self.bytecode[self.pc] if self.pc < len(self.bytecode) else 0, 
                             self.cycles):
                bp.hit_count += 1
                hit = True
        return hit
    
    def _signed_byte(self, b):
        return b - 256 if b > 127 else b
    
    def _execute_one(self):
        """Execute one instruction. Returns True if executed."""
        if self.halted or self.pc >= len(self.bytecode):
            return False
        
        op = self.bytecode[self.pc]
        
        if op == 0x00: self.halted = True; self.pc += 1
        elif op == 0x01: self.pc += 1
        elif op == 0x08: self.registers[self.bytecode[self.pc+1]] += 1; self.pc += 2
        elif op == 0x09: self.registers[self.bytecode[self.pc+1]] -= 1; self.pc += 2
        elif op == 0x0A: rd = self.bytecode[self.pc+1]; self.registers[rd] = ~self.registers[rd]; self.pc += 2
        elif op == 0x0B: rd = self.bytecode[self.pc+1]; self.registers[rd] = -self.registers[rd]; self.pc += 2
        elif op == 0x0C: self.sp -= 1; self.stack[self.sp] = self.registers[self.bytecode[self.pc+1]]; self.pc += 2
        elif op == 0x0D: self.registers[self.bytecode[self.pc+1]] = self.stack[self.sp]; self.sp += 1; self.pc += 2
        elif op == 0x18: self.registers[self.bytecode[self.pc+1]] = self._signed_byte(self.bytecode[self.pc+2]); self.pc += 3
        elif op == 0x19: self.registers[self.bytecode[self.pc+1]] += self._signed_byte(self.bytecode[self.pc+2]); self.pc += 3
        elif op == 0x20: self.registers[self.bytecode[self.pc+1]] = self.registers[self.bytecode[self.pc+2]] + self.registers[self.bytecode[self.pc+3]]; self.pc += 4
        elif op == 0x21: self.registers[self.bytecode[self.pc+1]] = self.registers[self.bytecode[self.pc+2]] - self.registers[self.bytecode[self.pc+3]]; self.pc += 4
        elif op == 0x22: self.registers[self.bytecode[self.pc+1]] = self.registers[self.bytecode[self.pc+2]] * self.registers[self.bytecode[self.pc+3]]; self.pc += 4
        elif op == 0x23:
            if self.registers[self.bytecode[self.pc+3]] != 0:
                self.registers[self.bytecode[self.pc+1]] = self.registers[self.bytecode[self.pc+2]] // self.registers[self.bytecode[self.pc+3]]
            self.pc += 4
        elif op == 0x2E: self.registers[self.bytecode[self.pc+1]] = 1 if self.registers[self.bytecode[self.pc+2]] > self.registers[self.bytecode[self.pc+3]] else 0; self.pc += 4
        elif op == 0x3A: self.registers[self.bytecode[self.pc+1]] = self.registers[self.bytecode[self.pc+2]]; self.pc += 4
        elif op == 0x3D:
            if self.registers[self.bytecode[self.pc+1]] != 0:
                self.pc += self._signed_byte(self.bytecode[self.pc+2])
            else:
                self.pc += 4
        elif op == 0x40:
            imm = self.bytecode[self.pc+2] | (self.bytecode[self.pc+3] << 8)
            if imm > 0x7FFF: imm -= 0x10000
            self.registers[self.bytecode[self.pc+1]] = imm; self.pc += 4
        else:
            self.pc += 1
        
        self.cycles += 1
        return True
    
    def step(self) -> Optional[DebugState]:
        """Execute one instruction and return state."""
        if not self._execute_one():
            return None
        state = self._save_state(self.bytecode[self.pc - 1] if self.pc > 0 else 0)
        self.history.append(state)
        return state
    
    def step_over(self) -> Optional[DebugState]:
        """Step over (same as step for FLUX — no function calls)."""
        return self.step()
    
    def run_until_break(self) -> List[DebugState]:
        """Run until a breakpoint is hit or program ends."""
        states = []
        for _ in range(100000):  # safety limit
            if self._check_breakpoints():
                break
            state = self.step()
            if state is None:
                break
            states.append(state)
        return states
    
    def run_all(self) -> List[DebugState]:
        """Run to completion."""
        states = []
        for _ in range(100000):
            state = self.step()
            if state is None:
                break
            states.append(state)
        return states
    
    def reverse(self) -> Optional[DebugState]:
        """Undo last step."""
        if len(self.history) < 2:
            return None
        self.history.pop()  # remove current
        prev = self.history[-1]
        # Restore state from previous snapshot
        self.pc = prev.pc
        self.registers = prev.registers.copy() + [0] * (64 - len(prev.registers))
        self.sp = prev.sp
        self.cycles = prev.cycles
        self.halted = prev.halted
        return prev
    
    def inspect(self) -> dict:
        """Return current state as dict."""
        return {
            "pc": self.pc,
            "cycles": self.cycles,
            "halted": self.halted,
            "registers": {f"R{i}": self.registers[i] for i in range(8)},
            "sp": self.sp,
            "breakpoints": len(self.breakpoints),
            "history_size": len(self.history),
        }


# ── Tests ──────────────────────────────────────────────

import unittest


class TestBreakpoint(unittest.TestCase):
    def test_pc_breakpoint(self):
        bp = Breakpoint(bp_type=BreakpointType.PC, value=5)
        self.assertTrue(bp.should_break(5, [0]*64, 0, 0))
        self.assertFalse(bp.should_break(4, [0]*64, 0, 0))
    
    def test_register_breakpoint(self):
        bp = Breakpoint(bp_type=BreakpointType.REGISTER, value=42, register=0)
        self.assertTrue(bp.should_break(0, [42] + [0]*63, 0, 0))
        self.assertFalse(bp.should_break(0, [41] + [0]*63, 0, 0))
    
    def test_op_breakpoint(self):
        bp = Breakpoint(bp_type=BreakpointType.OP, value=0x22)  # MUL
        self.assertTrue(bp.should_break(0, [0]*64, 0x22, 0))
    
    def test_cycle_breakpoint(self):
        bp = Breakpoint(bp_type=BreakpointType.CYCLE, value=100)
        self.assertFalse(bp.should_break(0, [0]*64, 0, 50))
        self.assertTrue(bp.should_break(0, [0]*64, 0, 100))
    
    def test_disabled_breakpoint(self):
        bp = Breakpoint(bp_type=BreakpointType.PC, value=5, enabled=False)
        self.assertFalse(bp.should_break(5, [0]*64, 0, 0))
    
    def test_register_gt(self):
        bp = Breakpoint(bp_type=BreakpointType.REGISTER, value=10, register=0, condition="gt")
        self.assertTrue(bp.should_break(0, [11]+[0]*63, 0, 0))
        self.assertFalse(bp.should_break(0, [10]+[0]*63, 0, 0))
    
    def test_hit_count(self):
        bp = Breakpoint(bp_type=BreakpointType.PC, value=5)
        bp.should_break(5, [0]*64, 0, 0)
        self.assertEqual(bp.hit_count, 1)


class TestDebugger(unittest.TestCase):
    def test_step_halt(self):
        dbg = FluxDebugger([0x00])
        state = dbg.step()
        self.assertTrue(dbg.halted)
    
    def test_step_movi(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.step()  # MOVI
        self.assertEqual(dbg.registers[0], 42)
    
    def test_step_add(self):
        dbg = FluxDebugger([0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00])
        for _ in range(4):
            dbg.step()
        self.assertEqual(dbg.registers[2], 30)
    
    def test_run_all(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        states = dbg.run_all()
        self.assertTrue(dbg.halted)
        self.assertEqual(dbg.registers[0], 42)
    
    def test_breakpoint_stops(self):
        dbg = FluxDebugger([0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00])
        dbg.add_breakpoint(Breakpoint(bp_type=BreakpointType.PC, value=6))  # Before ADD
        states = dbg.run_until_break()
        self.assertEqual(dbg.pc, 6)
    
    def test_reverse(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.step()  # MOVI
        dbg.step()  # HALT
        prev = dbg.reverse()
        self.assertIsNotNone(prev)
        self.assertFalse(dbg.halted)
    
    def test_inspect(self):
        dbg = FluxDebugger([0x18, 0, 42, 0x00])
        dbg.step()
        info = dbg.inspect()
        self.assertEqual(info["registers"]["R0"], 42)
    
    def test_remove_breakpoint(self):
        dbg = FluxDebugger([0x00])
        dbg.add_breakpoint(Breakpoint(bp_type=BreakpointType.PC, value=0))
        dbg.remove_breakpoint(0)
        self.assertEqual(len(dbg.breakpoints), 0)
    
    def test_factorial_with_breakpoints(self):
        # factorial(5) = 120
        dbg = FluxDebugger([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        # Break on MUL opcode
        dbg.add_breakpoint(Breakpoint(bp_type=BreakpointType.OP, value=0x22))
        states = dbg.run_until_break()
        self.assertGreater(len(states), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
