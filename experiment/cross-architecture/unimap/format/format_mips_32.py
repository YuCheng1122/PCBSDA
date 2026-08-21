import re

# ── registers ─────────────────────────────────────────────────────────────────

# Canonical map: ABI name → Rn
_REG_MAP = {
    'zero':'R0', 'at':'R1',
    'v0':'R2',  'v1':'R3',
    'a0':'R4',  'a1':'R5',  'a2':'R6',  'a3':'R7',
    't0':'R8',  't1':'R9',  't2':'R10', 't3':'R11',
    't4':'R12', 't5':'R13', 't6':'R14', 't7':'R15',
    's0':'R16', 's1':'R17', 's2':'R18', 's3':'R19',
    's4':'R20', 's5':'R21', 's6':'R22', 's7':'R23',
    't8':'R24', 't9':'R25',
    'k0':'R26', 'k1':'R27',
    'gp':'R28', 'sp':'R29', 'fp':'R30', 'ra':'R31',
    's8':'R30',  # alias for fp
}

# Numeric names r0-r31
_NUM_REGS = {f'r{i}': f'R{i}' for i in range(32)}
_REG_MAP.update(_NUM_REGS)

# FPU registers f0-f31
_FPU_REGS = {f'f{i}': f'F{i}' for i in range(32)}

# FPU condition registers fcc0-fcc7
_FCC_REGS = {f'fcc{i}': f'FCC{i}' for i in range(8)}

# Accumulator / special / FPU control
_SPECIAL = {'hi': 'HI', 'lo': 'LO', 'pc': 'PC', 'fcsr': 'FCSR', 'fir': 'FIR'}

ALL_NAMED = set(_REG_MAP) | set(_FPU_REGS) | set(_FCC_REGS) | set(_SPECIAL)

def _norm_reg(r):
    r = r.lower().strip()
    if r in _REG_MAP:   return _REG_MAP[r]
    if r in _FPU_REGS:  return _FPU_REGS[r]
    if r in _FCC_REGS:  return _FCC_REGS[r]
    if r in _SPECIAL:   return _SPECIAL[r]
    return r.upper()

def _is_reg(r):
    return r.lower().strip() in ALL_NAMED

# ── branch / jump mnemonics ───────────────────────────────────────────────────

_BRANCH_OPS = {
    'b','bal','beq','bne','bgez','bgtz','blez','bltz',
    'bgezal','bltzal','beqz','bnez',
    'bc1t','bc1f','bc1tl','bc1fl',
    'j','jal','jalr','jr',
}

# ── opcode normalizations ─────────────────────────────────────────────────────

# Merge variant opcodes to canonical base
_OP_MAP = {
    'beqz':'BEQ', 'bnez':'BNE',
    'addu':'ADD', 'addiu':'ADD', 'addi':'ADD',
    'subu':'SUB',
    'multu':'MULT', 'mulu':'MUL',
    'divu':'DIV',
    'sltu':'SLT', 'sltiu':'SLT', 'slti':'SLT',
    'xori':'XOR', 'andi':'AND', 'ori':'OR',
    'srav':'SRA', 'sllv':'SLL', 'srlv':'SRL',
    'lwl':'LW', 'lwr':'LW', 'lwc1':'LW', 'ldc1':'LW', 'ldxc1':'LW',
    'swl':'SW', 'swr':'SW', 'swc1':'SW', 'sdc1':'SW',
    'lhu':'LH', 'lbu':'LB',
    'maddu':'MADD',
    'seb':'SEB', 'seh':'SEH',
}

# ── directives to skip ────────────────────────────────────────────────────────

_DIRECTIVES = {'.data','.text','.extern','.globl','.align',
               '.ascii','.asciiz','.byte','.half','.word','.space','space'}

# ── helpers ───────────────────────────────────────────────────────────────────

_HEX_RE = re.compile(r'^-?0x[0-9a-fA-F]+$', re.IGNORECASE)
_DEC_RE = re.compile(r'^-?\d+$')

def _is_imm(s):
    s = s.strip()
    return bool(_HEX_RE.match(s) or _DEC_RE.match(s))

def _norm_imm(s):
    s = s.strip()
    return '-0' if s.startswith('-') else '0'

# ── token helpers ─────────────────────────────────────────────────────────────

def _is_opcode_token(token):
    """Opcode: starts with a letter, no parens/comma/dot (except float ops like add.D)."""
    t = token.lower()
    # Ghidra delay-slot marker: _nop, _addiu etc → strip leading _
    t = t.lstrip('_')
    if not t:
        return False
    if not re.match(r'^[a-z]', t):
        return False
    if any(c in t for c in '(),'):
        return False
    # register names are not opcodes
    if t in ALL_NAMED:
        return False
    # plain hex immediate
    if re.match(r'^-?0x[0-9a-f]+$', t) or re.match(r'^-?\d+$', t):
        return False
    return True

def _split_to_instructions(tokens):
    instructions = []
    current = []
    for t in tokens:
        if _is_opcode_token(t) and current:
            instructions.append(current)
            current = [t]
        else:
            if not current:
                current = [t]
            else:
                current.append(t)
    if current:
        instructions.append(current)
    return instructions

# ── operand normalizer ────────────────────────────────────────────────────────

def _norm_operand(token, is_branch=False):
    t = token.strip()

    # strip trailing comma
    trailing_comma = t.endswith(',')
    if trailing_comma:
        t = t[:-1]

    result = _norm_operand_inner(t, is_branch)
    if trailing_comma and result:
        result += ','
    return result


def _norm_operand_inner(t, is_branch):
    # register
    if _is_reg(t):
        return _norm_reg(t)

    # plain immediate or address
    if _is_imm(t):
        return '<FOO>' if is_branch else _norm_imm(t)

    # indexed: reg(reg)  e.g. a0(a1)  — ldxc1/sdxc1 format
    m_idx = re.match(r'^(\w+)\((\w+)\)$', t)
    if m_idx and _is_reg(m_idx.group(1)) and _is_reg(m_idx.group(2)):
        return '<OFF>' + _norm_reg(m_idx.group(1)) + _norm_reg(m_idx.group(2))

    # offset(reg)  e.g.  0x10(sp)  or  -0x7da4(gp)
    m_off = re.match(r'^(-?(?:0x[0-9a-fA-F]+|\d+))\((\w+)\)$', t)
    if m_off:
        reg = _norm_reg(m_off.group(2))
        return '<OFF>' + reg

    # double-paren:  0x0(reg)(reg)
    m_dbl = re.match(r'^.*\((\w+)\)\((\w+)\)$', t)
    if m_dbl:
        base = _norm_reg(m_dbl.group(1))
        idx  = _norm_reg(m_dbl.group(2))
        return '<OFF>' + base + idx

    # comma-joined pair:  reg,addr  e.g. "0x1,0x0040215c"
    if ',' in t:
        parts = t.split(',', 1)
        return _norm_operand_inner(parts[0], False) + ',' + _norm_operand_inner(parts[1], is_branch)

    return '<TAG>'

# ── main normalizer ───────────────────────────────────────────────────────────

_SKIP_INSNS = {'nop'}

def normalize_line(raw_line):
    raw_line = raw_line.strip()
    if not raw_line:
        return None

    # strip inline comments
    for comment_char in (';', '#'):
        if comment_char in raw_line:
            raw_line = raw_line[:raw_line.index(comment_char)]

    # Ghidra delay-slot prefix _  e.g.  '_nop', '_addiu sp,sp,0x20'
    # Strip leading _ from each token — we handle it in normalization
    tokens = raw_line.split()
    instructions = _split_to_instructions(tokens)

    parts = []
    for instr in instructions:
        normed = _norm_mips_instruction(instr)
        if normed:
            parts.append(normed)

    return ' '.join(parts) if parts else None


def _norm_mips_instruction(tokens):
    if not tokens:
        return None

    raw_op = tokens[0].lower().lstrip('_')  # strip Ghidra delay-slot marker

    if raw_op in _SKIP_INSNS:
        return None
    if raw_op in _DIRECTIVES:
        return None
    if not raw_op:
        return None

    # float opcodes: add.D, mul.S, cvt.d.W → keep as-is but uppercase
    # strip .condition suffix if present (e.g. movn.D → MOVN)
    canonical = _OP_MAP.get(raw_op, raw_op.upper())

    operands = tokens[1:]
    # Drop spurious single-letter type tokens Ghidra sometimes emits (e.g. "sub.D D f0,…")
    if operands and re.match(r'^[DWSFL]$', operands[0], re.IGNORECASE):
        operands = operands[1:]
    is_branch = raw_op in _BRANCH_OPS

    if not operands:
        return canonical

    # branch with single target
    if is_branch and len(operands) == 1:
        return canonical + '~' + _norm_operand(operands[0], is_branch=True)

    normed_ops = [_norm_operand(op) for op in operands]
    normed_ops = [o for o in normed_ops if o]

    return '~'.join([canonical] + normed_ops)
