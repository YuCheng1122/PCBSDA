import re

# ── registers ─────────────────────────────────────────────────────────────────

# General purpose r0-r15
_GPR = {f'r{i}' for i in range(16)} | {f'r{i}!' for i in range(16)}

# Named aliases
_ALIASES = {'sp', 'lr', 'pc', 'ip', 'sb', 'fp',
            'sp!', 'lr!', 'pc!'}

# NEON / VFP
_FLOAT = (
    {f'd{i}' for i in range(32)} |
    {f's{i}' for i in range(32)} |
    {f'q{i}' for i in range(16)}
)

# Coprocessor / system
_SYS = {'cpsr', 'spsr', 'fpscr', 'apsr'}

# Coprocessor names p0-p15 and coprocessor registers cr0-cr15
_CP_REGS = {f'p{i}' for i in range(16)} | {f'cr{i}' for i in range(16)} | {f'c{i}' for i in range(16)}

ALL_REGS = _GPR | _ALIASES | _FLOAT | _SYS | _CP_REGS

# Canonical mapping for named aliases → Rn
_ALIAS_MAP = {
    'sp': 'r13', 'sp!': 'r13!',
    'lr': 'r14', 'lr!': 'r14!',
    'pc': 'r15', 'pc!': 'r15!',
    'ip': 'r12',
    'sb': 'r9',
    'fp': 'r11',
}

# ── condition codes (suffixes on opcodes) ─────────────────────────────────────

_COND_CODES = {
    'eq','ne','cs','cc','mi','pl','vs','vc',
    'hi','ls','ge','lt','gt','le','al',
    'hs','lo',  # aliases for cs/cc
}

# ── branch / call mnemonics ───────────────────────────────────────────────────

_BRANCH_BASE = {
    'b','bl','bx','blx','bxj',
    'beq','bne','bcs','bcc','bmi','bpl','bvs','bvc',
    'bhi','bls','bge','blt','bgt','ble','bal',
    'bhs','blo',
}

# ── helpers ───────────────────────────────────────────────────────────────────

_HEX_RE = re.compile(r'^-?0x[0-9a-fA-F]+$', re.IGNORECASE)
_DEC_RE = re.compile(r'^-?\d+$')

def _is_imm(s):
    s = s.strip().lstrip('#')
    return bool(_HEX_RE.match(s) or _DEC_RE.match(s))

def _norm_imm(s):
    s = s.strip().lstrip('#')
    return '-0' if s.startswith('-') else '0'

def _norm_reg(r):
    """Uppercase and canonicalize a register name."""
    r = r.lower().strip()
    r = _ALIAS_MAP.get(r, r)
    return r.upper()

def _is_reg(r):
    return r.lower().strip().rstrip('!') in {x.rstrip('!') for x in ALL_REGS} or \
           r.lower().strip() in ALL_REGS

# ── condition/width suffix stripping ─────────────────────────────────────────

_WIDTH_SUFFIX = re.compile(r'\.(w|n)$', re.IGNORECASE)

# STM/LDM addressing-mode suffixes (not condition codes)
_STACK_MODES = {'ia','ib','da','db','fd','fa','ed','ea'}

# All ARM condition codes
_COND_SET = {'eq','ne','cs','cc','hs','lo','mi','pl','vs','vc','hi','ls','ge','lt','gt','le','al'}

def _strip_opcode(raw):
    """
    Strip condition-code suffix and .W/.N width specifier from opcode.
    Returns (base_opcode_uppercase, cond_code_uppercase_or_empty).

    Handles:
      addeq   → (ADD, EQ)
      ldreqs  → (LDRS, EQ)      # 's' update flag is part of base
      bleq    → (BL, EQ)
      stmdb   → (STMDB, '')     # 'db' is addressing mode, not cond
      stmdbeq → (STMDB, EQ)
      ldmiane → (LDMIA, NE)
    """
    op = _WIDTH_SUFFIX.sub('', raw).lower()

    # known branch ops — no cond stripping needed (already encoded in name)
    if op in _BRANCH_BASE:
        return op.upper(), ''

    # STM/LDM: base is stm/ldm + mode (2 chars) + optional cond (2 chars) + optional 's'
    stm_m = re.match(r'^(stm|ldm|push|pop)(ia|ib|da|db|fd|fa|ed|ea)?(eq|ne|cs|cc|hs|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le|al)?(s?)$', op)
    if stm_m:
        base = stm_m.group(1) + (stm_m.group(2) or '')
        cond = stm_m.group(3) or ''
        sfx  = stm_m.group(4) or ''
        return (base + sfx).upper(), cond.upper()

    # Generic: try stripping 2-char cond code from end.
    # Strategy: try all possible cond suffixes, prefer the one whose remaining
    # base is longer (more specific). Also handle trailing 's' (set-flags).
    #
    # Known pitfall: "movs" → base=mov, sflag='s', cond=''
    #                "movsne" → base=movs or mov, cond=ne
    #                "movvs" → base=mov, cond=vs  (vs is a real cond code)
    # We try cond stripping only if the remainder is ≥ 2 chars.

    # Minimum known ARM opcode lengths (to avoid stripping too aggressively)
    _MIN_BASE = 2

    for cond in sorted(_COND_SET, key=len, reverse=True):
        if op.endswith(cond):
            base = op[:-len(cond)]
            if len(base) < _MIN_BASE:
                continue
            # "movs" ends with "vs" leaving base "mo" (len 2) — reject if base
            # looks like it ends with 's' that is a set-flags marker, i.e. the
            # real base would be base[:-1] and base[-1]=='s' and base[:-1] is valid.
            # Simpler heuristic: reject if base ends with 's' AND base[:-1] >= 2 chars,
            # because then "movs" = base "mo" + stripped "vs" is wrong;
            # the correct parse is base "movs" + no cond.
            # We check: does op without the cond make sense as an opcode?
            # If the remaining base is only 2 chars AND it ends with a vowel or
            # single consonant, it's likely wrong — skip and fall through to 's' check.
            if len(base) == 2 and base[1] in 'aeiou':
                continue
            return base.upper(), cond.upper()

    # No cond found — keep full opcode (may include 's' set-flags suffix)
    return op.upper(), ''

# ── memory operand normalizer ─────────────────────────────────────────────────

def _norm_mem(inner):
    """
    Normalize expression inside [...].
    e.g.  sp,#0x8        → R13,0
          r3,r0,lsl #0x2 → R3,R0,LSL 0
          pc,r0,lsl #0x2 → R15,R0,LSL 0
          0x12928        → 0
    """
    # Handle shift embedded as separate space-joined pair inside [...]:
    # "r3,r0,lsl #0x2" — split by comma first, then handle "lsl #0x2" as one part
    parts = [p.strip() for p in inner.split(',')]
    result = []
    i = 0
    while i < len(parts):
        p = parts[i]
        p_low = p.lower().lstrip('#')
        # handle "lsl#0x2" (no space) as well as "lsl #0x2"
        p_spaced = re.sub(r'(lsl|lsr|asr|ror|rrx)(#)', r'\1 \2', p_low, flags=re.IGNORECASE)
        shift_kw = p_spaced.split()[0] if ' ' in p_spaced else p_spaced
        p_low = p_spaced  # use spaced version for further processing
        if shift_kw in ('lsl','lsr','asr','ror','rrx'):
            # may be "lsl #0x2" as one part or "lsl" + next-part is amount
            tokens_in = p_low.split()
            if len(tokens_in) >= 2:
                result.append(tokens_in[0].upper() + '~' + _norm_imm(tokens_in[1]))
            elif i + 1 < len(parts):
                result.append(p_low.upper() + '~' + _norm_imm(parts[i+1]))
                i += 1
            else:
                result.append(p_low.upper())
        elif p_low.rstrip('!') in {x.rstrip('!') for x in ALL_REGS} or p_low in ALL_REGS:
            result.append(_norm_reg(p))
        elif _is_imm(p):
            result.append(_norm_imm(p))
        else:
            result.append('<TAG>')
        i += 1
    return ','.join(result)

# ── register list expander ────────────────────────────────────────────────────

def _expand_reglist(token):
    """
    Expand {r0,r1-r3,lr} into individual register names.
    Returns list of register name strings (uppercased, canonicalized).
    """
    inner = token.strip().strip('{}').strip()
    regs = []
    for part in inner.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-', 1)
            lo = _alias_to_num(lo.strip())
            hi = _alias_to_num(hi.strip())
            if lo is not None and hi is not None:
                for n in range(lo, hi + 1):
                    regs.append(f'R{n}')
        else:
            r = _norm_reg(part) if _is_reg(part) else part.upper()
            regs.append(r)
    return regs

def _alias_to_num(r):
    r = r.lower()
    m = re.match(r'^r(\d+)$', r)
    if m:
        return int(m.group(1))
    num = {'sp':13,'lr':14,'pc':15,'ip':12,'sb':9,'fp':11}.get(r)
    return num

# ── instruction splitter ──────────────────────────────────────────────────────

# ARM opcodes are lowercase alpha (possibly with digits like ldm, stm variants)
_SHIFT_KEYWORDS = {'lsl','lsr','asr','ror','rrx'}

def _is_opcode_token(token):
    t = token.lower()
    # starts with alpha, may contain alpha+digits
    if not re.match(r'^[a-z]', t):
        return False
    # shift keywords are operand modifiers, not opcodes
    if t in _SHIFT_KEYWORDS:
        return False
    # register names are not opcodes
    if t.rstrip('!') in {x.rstrip('!') for x in ALL_REGS}:
        return False
    # pure hex immediate
    if re.match(r'^0x[0-9a-f]+$', t):
        return False
    # bracket or special chars → not opcode
    if any(c in t for c in '[]{}#,'):
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
    """Normalize a single ARM operand token."""
    t = token.strip()

    # register
    tl = t.lower().rstrip('!')
    if tl in {x.rstrip('!') for x in ALL_REGS} or t.lower() in ALL_REGS:
        return _norm_reg(t)

    # register list  {r0,r1,...}
    if t.startswith('{'):
        regs = _expand_reglist(t)
        return '{' + ','.join(regs) + '}'

    # immediate  #0x1c  or  #-0x4  or  #0
    if t.startswith('#'):
        return _norm_imm(t)

    # absolute address or plain hex/dec immediate
    if _is_imm(t):
        if is_branch:
            return '<FOO>'
        return _norm_imm(t)

    # REG,[...]  or  REG!,[...]  — base register before a bracket
    bk_idx = t.find(',[')
    if bk_idx > 0:
        prefix = t[:bk_idx]
        bracket = t[bk_idx+1:]  # starts with '['
        nr = _norm_reg(prefix) if _is_reg(prefix.rstrip('!')) else prefix.upper()
        return nr + ',' + _norm_operand(bracket)

    # memory [base, offset]  or post-index [base],#imm  or [base]!
    if t.startswith('['):
        exclaim = t.endswith('!')
        t_body = t.rstrip('!')
        # post-index: [reg],#imm  — ']' not at end
        post_m = re.match(r'^\[([^\]]+)\],(.+)$', t_body)
        if post_m:
            body    = post_m.group(1)
            post_op = post_m.group(2).lstrip('#')
            post_norm = _norm_imm(post_op) if _is_imm(post_op) else _norm_reg(post_op) if _is_reg(post_op) else post_op.upper()
            return '[' + _norm_mem(body) + '],' + post_norm
        body = t_body.rstrip(']').lstrip('[')
        return '[' + _norm_mem(body) + (']!' if exclaim else ']')

    # shift keyword standalone
    if t.lower() in ('lsl','lsr','asr','ror','rrx'):
        return t.upper()

    # =address  (PC-relative load pseudo)
    if t.startswith('='):
        inner = t[1:]
        if _is_imm(inner):
            return '=0'
        return '=<FOO>'

    # comma-joined pair
    if ',' in t:
        parts = t.split(',', 1)
        return ','.join(_norm_operand(p) for p in parts)

    return '<TAG>'

# ── main normalizer ───────────────────────────────────────────────────────────

_SKIP_INSNS = {'nop', 'nop.w'}
_DIRECTIVES = {'code16','code32','export','align','dcd','dcb','import',
               'thumb','arm','ltorg','end'}

def normalize_line(raw_line):
    raw_line = raw_line.strip()
    if not raw_line:
        return None

    if ';' in raw_line:
        raw_line = raw_line[:raw_line.index(';')]

    tokens = _rejoin_arm_tokens(raw_line.split())
    instructions = _split_to_instructions(tokens)

    parts = []
    for instr in instructions:
        normed = _norm_arm_instruction(instr)
        if normed:
            parts.append(normed)

    return ' '.join(parts) if parts else None


def _rejoin_arm_tokens(tokens):
    """
    Merge multi-token operands:
      '[sp,' '#0x8]'  → '[sp,#0x8]'
      '[r3,' 'r0,' 'lsl' '#0x2]' → '[r3,r0,lsl #0x2]'
      '{r0,' 'r1,' 'r2}' → '{r0,r1,r2}'
    """
    out = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        # open bracket without close
        if ('[' in t and ']' not in t) or ('{' in t and '}' not in t):
            close = ']' if '[' in t else '}'
            fragment = t
            i += 1
            while i < len(tokens) and close not in tokens[i]:
                fragment += ' ' + tokens[i]   # preserve space (e.g. "lsl #0x2")
                i += 1
            if i < len(tokens):
                fragment += ' ' + tokens[i]   # closing token may be "#0x2]"
                i += 1
            out.append(fragment)
            continue
        out.append(t)
        i += 1
    return out


def _norm_arm_instruction(tokens):
    if not tokens:
        return None

    raw_op = tokens[0].lower()

    if raw_op in _SKIP_INSNS:
        return None
    if raw_op in _DIRECTIVES:
        return None

    base_op, cond = _strip_opcode(raw_op)

    if not base_op:
        return None

    operands = tokens[1:]
    is_branch = base_op.upper() in {b.upper() for b in _BRANCH_BASE} or \
                raw_op.startswith('b')

    # STM/LDM with register list: stmdb sp!,{r4,lr}
    if re.match(r'^(STM|LDM|PUSH|POP)', base_op.upper()):
        return _norm_stm_ldm(base_op, cond, operands)

    # branch: single address operand
    if is_branch and len(operands) == 1:
        tgt = _norm_operand(operands[0], is_branch=True)
        op_str = base_op + ('~' + cond if cond else '')
        return op_str + '~' + tgt

    normed_ops = []
    j = 0
    while j < len(operands):
        op = operands[j]
        op_low = op.lower()

        # shift keyword: merge with next token (the amount)
        if op_low in _SHIFT_KEYWORDS:
            if j + 1 < len(operands):
                normed_ops.append(op.upper() + '~' + _norm_operand(operands[j+1]))
                j += 2
            else:
                normed_ops.append(op.upper())
                j += 1
            continue

        # trailing-comma operand "r4," followed by shift keyword
        # e.g.  mov r3,r4, lsl #0x2  →  tokens: r3,r4, | lsl | #0x2
        if op.endswith(',') and j + 1 < len(operands) and operands[j+1].lower() in _SHIFT_KEYWORDS:
            # attach the shift to this operand
            base = _norm_operand(op[:-1])
            shift_kw = operands[j+1].upper()
            if j + 2 < len(operands):
                shift_amt = _norm_operand(operands[j+2])
                normed_ops.append(base + ',' + shift_kw + '~' + shift_amt)
                j += 3
            else:
                normed_ops.append(base + ',' + shift_kw)
                j += 2
            continue

        n = _norm_operand(op, is_branch=False)
        if n is not None:
            normed_ops.append(n)
        j += 1

    op_str = base_op + ('~' + cond if cond else '')
    return '~'.join([op_str] + normed_ops)


def _norm_stm_ldm(base_op, cond, operands):
    """
    Normalize STM/LDM/PUSH/POP which may have base reg + register list.
    Output one token per register (like the reference formatter does).
    """
    op_str = base_op + ('~' + cond if cond else '')

    # collect base register and register list
    base_reg = None
    reg_list = []
    for op in operands:
        if op.startswith('{'):
            reg_list = _expand_reglist(op)
        elif _is_reg(op) or op.lower().rstrip('!') in {x.rstrip('!') for x in ALL_REGS}:
            base_reg = _norm_reg(op)
        elif ',' in op and '{' in op:
            # base_reg,{list} merged
            idx = op.index(',{')
            base_reg = _norm_reg(op[:idx])
            reg_list = _expand_reglist(op[idx+1:])
        elif ',' in op:
            # base_reg, list split oddly
            parts = op.split(',', 1)
            base_reg = _norm_reg(parts[0])

    if not reg_list:
        # fallback: just emit raw normalized operands
        normed = [_norm_operand(o) for o in operands]
        return '~'.join([op_str] + [n for n in normed if n])

    results = []
    for reg in reg_list:
        if base_reg:
            results.append(op_str + '~' + base_reg + '~' + reg)
        else:
            results.append(op_str + '~' + reg)
    return ' '.join(results)
