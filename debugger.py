"""
FLUX Step Debugger — debug FLUX bytecodes across all runtimes.

Features:
- Step through bytecodes one instruction at a time
- Set breakpoints on PC values or labels
- Inspect registers, stack, memory at any point
- Conditional breakpoints (register value checks)
- Watch expressions (register and memory watches)
- Reverse step (undo last instruction)
- Step-over (skip over CALL bodies)
- Step-out (continue until RET)
- Execution trace recording
- Memory inspection and modification
- Stack inspection
- Breakpoint management (set/remove/disable by PC or label)
"""
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple, Callable, Any
from enum import Enum, auto


class BreakpointType(Enum):
    PC = auto()       # Break at specific PC
    REGISTER = auto() # Break when register matches condition
    OP = auto()       # Break on specific opcode
    CYCLE = auto()    # Break after N cycles
    LABEL = auto()    # Break at labeled PC


@dataclass
class Breakpoint:
    bp_type: BreakpointType
    value: int = 0
    register: int = 0
    condition: str = "eq"  # eq, ne, gt, lt, gte, lte
    enabled: bool = True
    hit_count: int = 0
    label: str = ""  # for LABEL type breakpoints
    id: int = 0
    
    def should_break(self, pc: int, registers: list, op: int, cycles: int,
                     labels: Optional[Dict[str, int]] = None) -> bool:
        if not self.enabled:
            return False
        self.hit_count += 1
        
        if self.bp_type == BreakpointType.PC:
            return pc == self.value
        elif self.bp_type == BreakpointType.LABEL:
            if labels and self.label in labels:
                return pc == labels[self.label]
            return False
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
    call_depth: int = 0


@dataclass
class WatchResult:
    """Result of evaluating a watch expression."""
    name: str
    value: Any
    changed: bool = False
    watch_type: str = "register"  # register, memory, custom


@dataclass
class TraceEntry:
    """Single entry in execution trace."""
    cycle: int
    pc: int
    op: int
    op_name: str
    registers_snapshot: Dict[str, int]
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
        0x48: "CALL", 0x49: "RET",
    }
    
    def __init__(self, bytecode: list):
        self.bytecode = bytes(bytecode)
        self.registers = [0] * 64
        self.memory = [0] * 65536  # 64KB addressable memory
        self.stack = [0] * 4096
        self.sp = 4096
        self.pc = 0
        self.cycles = 0
        self.halted = False
        self.breakpoints: List[Breakpoint] = []
        self.history: List[DebugState] = []
        self.labels: Dict[str, int] = {}  # name -> pc
        self._next_bp_id = 1
        
        # Call stack for CALL/RET tracking
        self.call_stack: List[int] = []  # return addresses
        self.call_depth = 0
        
        # Watch expressions
        self.watches: Dict[str, Callable[[FluxDebugger], Any]] = {}
        self._prev_watch_values: Dict[str, Any] = {}
        
        # Execution trace recording
        self.trace: List[TraceEntry] = []
        self.trace_enabled = False
        self.trace_max_entries = 10000
    
    # ── Label management ────────────────────────────────────────────────
    
    def add_label(self, name: str, pc: int):
        """Add a named label for a PC address."""
        self.labels[name] = pc
    
    def resolve_label(self, name: str) -> Optional[int]:
        """Resolve a label name to its PC address."""
        return self.labels.get(name)
    
    # ── Breakpoint management ───────────────────────────────────────────
    
    def add_breakpoint(self, bp: Breakpoint) -> int:
        """Add a breakpoint and return its ID."""
        bp.id = self._next_bp_id
        self._next_bp_id += 1
        self.breakpoints.append(bp)
        return bp.id
    
    def add_pc_breakpoint(self, pc: int) -> int:
        """Convenience: add a breakpoint at a specific PC."""
        return self.add_breakpoint(Breakpoint(bp_type=BreakpointType.PC, value=pc))
    
    def add_label_breakpoint(self, label: str) -> int:
        """Convenience: add a breakpoint at a labeled PC."""
        return self.add_breakpoint(Breakpoint(bp_type=BreakpointType.LABEL, label=label))
    
    def add_opcode_breakpoint(self, opcode: int) -> int:
        """Convenience: break when a specific opcode is encountered."""
        return self.add_breakpoint(Breakpoint(bp_type=BreakpointType.OP, value=opcode))
    
    def add_cycle_breakpoint(self, cycle: int) -> int:
        """Convenience: break after N cycles."""
        return self.add_breakpoint(Breakpoint(bp_type=BreakpointType.CYCLE, value=cycle))
    
    def add_conditional_breakpoint(self, register: int, value: int,
                                    condition: str = "eq") -> int:
        """Convenience: break when register matches condition."""
        return self.add_breakpoint(Breakpoint(
            bp_type=BreakpointType.REGISTER,
            value=value, register=register, condition=condition,
        ))
    
    def remove_breakpoint(self, index_or_id) -> bool:
        """Remove breakpoint by index or ID. Returns True if removed."""
        if isinstance(index_or_id, int):
            # Try by ID first
            for i, bp in enumerate(self.breakpoints):
                if bp.id == index_or_id:
                    self.breakpoints.pop(i)
                    return True
            # Fall back to index
            if 0 <= index_or_id < len(self.breakpoints):
                self.breakpoints.pop(index_or_id)
                return True
        return False
    
    def disable_breakpoint(self, bp_id: int) -> bool:
        """Disable a breakpoint by ID. Returns True if found."""
        for bp in self.breakpoints:
            if bp.id == bp_id:
                bp.enabled = False
                return True
        return False
    
    def enable_breakpoint(self, bp_id: int) -> bool:
        """Enable a breakpoint by ID. Returns True if found."""
        for bp in self.breakpoints:
            if bp.id == bp_id:
                bp.enabled = True
                return True
        return False
    
    def get_breakpoint(self, bp_id: int) -> Optional[Breakpoint]:
        """Get a breakpoint by ID."""
        for bp in self.breakpoints:
            if bp.id == bp_id:
                return bp
        return None
    
    def list_breakpoints(self) -> List[dict]:
        """List all breakpoints as dicts."""
        return [
            {
                "id": bp.id,
                "type": bp.bp_type.name,
                "enabled": bp.enabled,
                "hit_count": bp.hit_count,
                "value": bp.value,
                "label": bp.label,
                "register": bp.register,
                "condition": bp.condition,
            }
            for bp in self.breakpoints
        ]
    
    def clear_breakpoints(self):
        """Remove all breakpoints."""
        self.breakpoints.clear()
    
    # ── Watch expressions ───────────────────────────────────────────────
    
    def add_watch(self, name: str, fn: Callable):
        """Add a watch expression. fn receives the debugger and returns a value."""
        self.watches[name] = fn
    
    def add_register_watch(self, name: str, register: int):
        """Watch a register value."""
        self.watches[name] = lambda dbg, r=register: dbg.registers[r]
    
    def add_memory_watch(self, name: str, address: int):
        """Watch a memory location."""
        self.watches[name] = lambda dbg, addr=address: dbg.memory[addr]
    
    def remove_watch(self, name: str):
        """Remove a watch expression."""
        if name in self.watches:
            del self.watches[name]
            self._prev_watch_values.pop(name, None)
    
    def evaluate_watches(self) -> List[WatchResult]:
        """Evaluate all watches and return results with change detection."""
        results = []
        for name, fn in self.watches.items():
            try:
                value = fn(self)
                prev = self._prev_watch_values.get(name)
                changed = prev is not None and value != prev
                self._prev_watch_values[name] = value
                results.append(WatchResult(
                    name=name, value=value, changed=changed,
                    watch_type="register" if "reg" in name.lower() else "memory" if "mem" in name.lower() else "custom",
                ))
            except Exception:
                results.append(WatchResult(name=name, value=None, changed=False, watch_type="custom"))
        return results
    
    # ── Execution trace ─────────────────────────────────────────────────
    
    def enable_trace(self, max_entries: int = 10000):
        """Enable execution trace recording."""
        self.trace_enabled = True
        self.trace_max_entries = max_entries
    
    def disable_trace(self):
        """Disable execution trace recording."""
        self.trace_enabled = False
    
    def clear_trace(self):
        """Clear recorded trace."""
        self.trace.clear()
    
    def get_trace(self) -> List[dict]:
        """Get trace entries as list of dicts."""
        return [asdict(e) for e in self.trace]
    
    def get_trace_json(self) -> str:
        """Get trace as JSON string."""
        return json.dumps([asdict(e) for e in self.trace], indent=2)
    
    # ── Memory inspection and modification ──────────────────────────────
    
    def read_memory(self, address: int, size: int = 1) -> List[int]:
        """Read memory at address. Returns list of bytes."""
        if address < 0 or address + size > len(self.memory):
            return [0] * size
        return self.memory[address:address + size]
    
    def write_memory(self, address: int, data: List[int]):
        """Write data to memory at address."""
        for i, byte in enumerate(data):
            if 0 <= address + i < len(self.memory):
                self.memory[address + i] = byte & 0xFF
    
    def read_memory_word(self, address: int) -> int:
        """Read a 16-bit little-endian word from memory."""
        lo = self.read_memory(address, 1)[0]
        hi = self.read_memory(address + 1, 1)[0]
        return lo | (hi << 8)
    
    def write_memory_word(self, address: int, value: int):
        """Write a 16-bit little-endian word to memory."""
        self.write_memory(address, [value & 0xFF, (value >> 8) & 0xFF])
    
    def memory_dump(self, start: int, end: int) -> List[Tuple[int, int]]:
        """Dump memory range as list of (address, value) tuples."""
        start = max(0, start)
        end = min(len(self.memory), end)
        return [(addr, self.memory[addr]) for addr in range(start, end)]
    
    # ── Stack inspection ────────────────────────────────────────────────
    
    def get_stack(self, count: int = 16) -> List[Tuple[int, int]]:
        """Get top `count` stack entries as (offset, value) tuples."""
        result = []
        for i in range(min(count, 4096 - self.sp)):
            addr = self.sp + i
            if addr < 4096:
                result.append((i, self.stack[addr]))
        return result
    
    def get_stack_depth(self) -> int:
        """Get current stack depth (number of items pushed)."""
        return 4096 - self.sp
    
    def get_call_stack(self) -> List[int]:
        """Get the call stack (return addresses)."""
        return list(self.call_stack)
    
    # ── State inspection ────────────────────────────────────────────────
    
    def _save_state(self, op: int, note: str = "") -> DebugState:
        return DebugState(
            pc=self.pc,
            op=op,
            op_name=self.OP_NAMES.get(op, f"UNKNOWN({op:#x})"),
            registers=self.registers[:16].copy(),
            stack=self.stack[max(self.sp, 0):min(self.sp + 16, 4096)].copy() if self.sp < 4096 else [],
            sp=self.sp,
            cycles=self.cycles,
            halted=self.halted,
            note=note,
            call_depth=self.call_depth,
        )
    
    def _check_breakpoints(self) -> bool:
        hit = False
        for bp in self.breakpoints:
            if bp.should_break(
                self.pc, self.registers,
                self.bytecode[self.pc] if self.pc < len(self.bytecode) else 0,
                self.cycles,
                self.labels,
            ):
                hit = True
        return hit
    
    def _record_trace(self, op: int, note: str = ""):
        """Record a trace entry if tracing is enabled."""
        if self.trace_enabled and len(self.trace) < self.trace_max_entries:
            regs_snapshot = {f"R{i}": self.registers[i] for i in range(8)}
            self.trace.append(TraceEntry(
                cycle=self.cycles,
                pc=self.pc,
                op=op,
                op_name=self.OP_NAMES.get(op, f"0x{op:02x}"),
                registers_snapshot=regs_snapshot,
                note=note,
            ))
    
    def _signed_byte(self, b):
        return b - 256 if b > 127 else b
    
    def _execute_one(self) -> int:
        """Execute one instruction. Returns the opcode executed, or -1 if halted/ended."""
        if self.halted or self.pc >= len(self.bytecode):
            return -1
        
        op = self.bytecode[self.pc]
        self._record_trace(op)
        
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
        elif op == 0x24:
            if self.registers[self.bytecode[self.pc+3]] != 0:
                self.registers[self.bytecode[self.pc+1]] = self.registers[self.bytecode[self.pc+2]] % self.registers[self.bytecode[self.pc+3]]
            self.pc += 4
        elif op == 0x25: self.registers[self.bytecode[self.pc+1]] = self.registers[self.bytecode[self.pc+2]] & self.registers[self.bytecode[self.pc+3]]; self.pc += 4
        elif op == 0x26: self.registers[self.bytecode[self.pc+1]] = self.registers[self.bytecode[self.pc+2]] | self.registers[self.bytecode[self.pc+3]]; self.pc += 4
        elif op == 0x27: self.registers[self.bytecode[self.pc+1]] = self.registers[self.bytecode[self.pc+2]] ^ self.registers[self.bytecode[self.pc+3]]; self.pc += 4
        elif op == 0x2C:
            self.registers[self.bytecode[self.pc+1]] = 1 if self.registers[self.bytecode[self.pc+2]] == self.registers[self.bytecode[self.pc+3]] else 0
            self.pc += 4
        elif op == 0x2D:
            self.registers[self.bytecode[self.pc+1]] = 1 if self.registers[self.bytecode[self.pc+2]] < self.registers[self.bytecode[self.pc+3]] else 0
            self.pc += 4
        elif op == 0x2E: self.registers[self.bytecode[self.pc+1]] = 1 if self.registers[self.bytecode[self.pc+2]] > self.registers[self.bytecode[self.pc+3]] else 0; self.pc += 4
        elif op == 0x2F:
            self.registers[self.bytecode[self.pc+1]] = 1 if self.registers[self.bytecode[self.pc+2]] != self.registers[self.bytecode[self.pc+3]] else 0
            self.pc += 4
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
        elif op == 0x48:  # CALL
            if self.pc + 1 < len(self.bytecode):
                target = self.bytecode[self.pc+1]
                self.call_stack.append(self.pc + 2)  # return address
                self.call_depth += 1
                self.pc = target
        elif op == 0x49:  # RET
            if self.call_stack:
                self.pc = self.call_stack.pop()
                self.call_depth = max(0, self.call_depth - 1)
            else:
                self.pc += 1
        else:
            self.pc += 1
        
        self.cycles += 1
        return op
    
    # ── Step execution ──────────────────────────────────────────────────
    
    def step(self) -> Optional[DebugState]:
        """Execute one instruction and return state."""
        op = self._execute_one()
        if op < 0:
            return None
        state = self._save_state(op)
        self.history.append(state)
        return state
    
    def step_over(self) -> Optional[DebugState]:
        """Step over: if current instruction is CALL, run until return."""
        if self.halted or self.pc >= len(self.bytecode):
            return None
        
        op = self.bytecode[self.pc]
        if op == 0x48:  # CALL
            # Save current call depth
            depth = self.call_depth
            # Step into the call
            state = self.step()
            if state is None:
                return None
            # Keep stepping until we return to the same or lower depth
            while self.call_depth > depth and not self.halted:
                state = self.step()
                if state is None:
                    break
            return state
        else:
            return self.step()
    
    def step_out(self) -> Optional[DebugState]:
        """Step out: continue until the current function returns."""
        if self.halted:
            return None
        if self.call_depth == 0:
            # Not in a function, just step
            return self.step()
        
        target_depth = self.call_depth - 1
        state = None
        while self.call_depth > target_depth and not self.halted:
            state = self.step()
            if state is None:
                break
        return state
    
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
            "stack_depth": self.get_stack_depth(),
            "call_depth": self.call_depth,
            "breakpoints": len(self.breakpoints),
            "history_size": len(self.history),
            "trace_size": len(self.trace),
            "labels": dict(self.labels),
        }
    
    def disassemble_at(self, pc: int) -> Optional[dict]:
        """Disassemble instruction at given PC."""
        if pc < 0 or pc >= len(self.bytecode):
            return None
        op = self.bytecode[pc]
        name = self.OP_NAMES.get(op, f"0x{op:02x}")
        
        # Find label for this PC
        label = ""
        for lname, lpc in self.labels.items():
            if lpc == pc:
                label = lname
                break
        
        result = {"pc": pc, "opcode": op, "name": name, "label": label}
        
        # Decode operands
        if op in (0x08, 0x09, 0x0A, 0x0B, 0x0D) and pc + 1 < len(self.bytecode):
            result["operands"] = f"R{self.bytecode[pc+1]}"
        elif op in (0x18, 0x19, 0x1A) and pc + 2 < len(self.bytecode):
            result["operands"] = f"R{self.bytecode[pc+1]}, {self._signed_byte(self.bytecode[pc+2])}"
        elif op in (0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                     0x2C, 0x2D, 0x2E, 0x2F, 0x3A) and pc + 3 < len(self.bytecode):
            result["operands"] = f"R{self.bytecode[pc+1]}, R{self.bytecode[pc+2]}, R{self.bytecode[pc+3]}"
        elif op == 0x40 and pc + 3 < len(self.bytecode):
            imm = self.bytecode[pc+2] | (self.bytecode[pc+3] << 8)
            if imm > 0x7FFF: imm -= 0x10000
            result["operands"] = f"R{self.bytecode[pc+1]}, {imm}"
        elif op == 0x48 and pc + 1 < len(self.bytecode):
            target = self.bytecode[pc+1]
            target_label = ""
            for lname, lpc in self.labels.items():
                if lpc == target:
                    target_label = f" ({lname})"
                    break
            result["operands"] = f"{target}{target_label}"
        
        return result
    
    def disassemble_range(self, start: int, end: int) -> List[dict]:
        """Disassemble a range of instructions."""
        result = []
        pc = start
        while pc < end and pc < len(self.bytecode):
            d = self.disassemble_at(pc)
            if d:
                result.append(d)
                # Advance PC based on opcode size
                op = d["opcode"]
                if op in (0x00, 0x01, 0x49):
                    pc += 1
                elif op in (0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x48):
                    pc += 2
                elif op in (0x18, 0x19, 0x1A):
                    pc += 3
                elif op in (0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                             0x2C, 0x2D, 0x2E, 0x2F, 0x3A, 0x40):
                    pc += 4
                else:
                    pc += 1
            else:
                break
        return result


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
