# flux-debugger

> Step debugger for FLUX bytecodes with breakpoints, conditional watches, reverse execution, and state inspection.

## What This Is

`flux-debugger` is a Python module providing an **interactive step debugger** for FLUX bytecode programs — it supports four breakpoint types (PC, register condition, opcode, cycle count), watch expressions, reverse-step undo, and comprehensive state inspection.

## Role in the FLUX Ecosystem

When things go wrong, the debugger is the scalpel:

- **`flux-timeline`** provides batch tracing; debugger provides interactive stepping
- **`flux-profiler`** finds slow code; debugger finds broken code
- **`flux-coverage`** shows untested paths; debugger walks through them one step at a time
- **`flux-decompiler`** shows what the code should do; debugger shows what it actually does
- **`flux-signatures`** identifies patterns; debugger verifies pattern behavior
- **`flux-disasm`** disassembles; debugger disassembles *and executes*

## Key Features

| Feature | Description |
|---------|-------------|
| **4 Breakpoint Types** | PC address, register condition (eq/ne/gt/lt/gte/lte), opcode, cycle count |
| **Conditional Breakpoints** | Break when R3 > 42, when MUL is encountered, after 100 cycles |
| **Step/Step-Over** | Execute one instruction at a time |
| **Reverse Step** | Undo last instruction (restores full VM state) |
| **Run Until Break** | Execute until breakpoint hit or program ends |
| **Run All** | Execute to completion, returning all states |
| **State Inspection** | PC, registers, stack, SP, cycles, breakpoint count, history size |
| **30+ Opcodes** | Full FLUX ISA execution support |

## Quick Start

```python
from flux_debugger import FluxDebugger, Breakpoint, BreakpointType

dbg = FluxDebugger([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0,
                     0x09, 0, 0x3D, 0, -6, 0, 0x00])  # factorial(5)

# Set breakpoints
dbg.add_breakpoint(Breakpoint(BreakpointType.PC, value=6))     # before MUL
dbg.add_breakpoint(Breakpoint(BreakpointType.OP, value=0x22))  # any MUL
dbg.add_breakpoint(Breakpoint(BreakpointType.REGISTER, value=0,
                               register=0, condition="eq"))    # when R0==0

# Run until break
states = dbg.run_until_break()
print(f"Stopped at PC={dbg.pc}")

# Step interactively
while True:
    state = dbg.step()
    if state is None:
        break
    print(f"[{state.cycles}] PC={state.pc} {state.op_name} R0={state.registers[0]}")

# Reverse a step
prev = dbg.reverse()

# Inspect
info = dbg.inspect()
print(info)
```

## Running Tests

```bash
python -m pytest tests/ -v
# or
python debugger.py
```

## Related Fleet Repos

- [`flux-timeline`](https://github.com/SuperInstance/flux-timeline) — Batch execution tracing
- [`flux-profiler`](https://github.com/SuperInstance/flux-profiler) — Performance profiling
- [`flux-coverage`](https://github.com/SuperInstance/flux-coverage) — Code coverage
- [`flux-decompiler`](https://github.com/SuperInstance/flux-decompiler) — Bytecode to assembly
- [`flux-disasm`](https://github.com/SuperInstance/flux-disasm) — C disassembler

## License

Part of the [SuperInstance](https://github.com/SuperInstance) FLUX fleet.
