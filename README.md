# FLUX Debugger — Step Debugger with Breakpoints

Debug FLUX bytecodes across all runtimes with a Python reference debugger.

## Features
- **Step**: Execute one instruction, inspect state
- **Breakpoints**: PC, register, opcode, cycle count
- **Conditional breakpoints**: eq, ne, gt, lt, gte, lte
- **Reverse step**: Undo last instruction
- **Watch expressions**: Custom Python callables
- **State inspection**: registers, stack, PC, cycles

## Usage

```python
from debugger import FluxDebugger, Breakpoint, BreakpointType

dbg = FluxDebugger([0x18, 0, 42, 0x00])
dbg.add_breakpoint(Breakpoint(bp_type=BreakpointType.REGISTER, value=42, register=0))
states = dbg.run_until_break()
print(dbg.inspect())
```

16 tests passing.
