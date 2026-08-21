import re

# ── registers ─────────────────────────────────────────────────────────────────

_GPR64  = {'RAX','RBX','RCX','RDX','RSP','RBP','RSI','RDI','RIP',
            'R8','R9','R10','R11','R12','R13','R14','R15'}
_GPR32  = {'EAX','EBX','ECX','EDX','ESP','EBP','ESI','EDI',
            'R8D','R9D','R10D','R11D','R12D','R13D','R14D','R15D'}
_GPR16  = {'AX','BX','CX','DX','SP','BP','SI','DI',
            'R8W','R9W','R10W','R11W','R12W','R13W','R14W','R15W'}
_GPR8   = {'AH','AL','BH','BL','CH','CL','DH','DL',
           'SPL','BPL','SIL','DIL',
           'R8B','R9B','R10B','R11B','R12B','R13B','R14B','R15B'}
_SEG    = {'CS','DS','ES','FS','GS','SS'}
_CTRL   = {f'CR{i}' for i in range(16)}
_DBG    = {f'DR{i}' for i in range(8)}
_BND    = {'BND0','BND1','BND2','BND3','BNDCFG','BNDCFU','BNDSTATUS'}
_ST     = {f'ST({i})' for i in range(8)} | {f'ST{i}' for i in range(8)}
_XMM    = {f'XMM{i}' for i in range(32)}
_YMM    = {f'YMM{i}' for i in range(32)}
_ZMM    = {f'ZMM{i}' for i in range(32)}

ALL_REGS = (_GPR64 | _GPR32 | _GPR16 | _GPR8 |
            _SEG | _CTRL | _DBG | _BND | _ST |
            _XMM | _YMM | _ZMM)

# ── immediates ────────────────────────────────────────────────────────────────

_HEX_RE = re.compile(r'^-?0x[0-9a-fA-F]+$', re.IGNORECASE)
_DEC_RE = re.compile(r'^-?\d+$')

def _is_imm(s):
    s = s.strip()
    return bool(_HEX_RE.match(s) or _DEC_RE.match(s))

def _norm_imm(s):
    return '-0' if s.strip().startswith('-') else '0'

# ── memory size qualifiers ────────────────────────────────────────────────────

_PTR_SIZES = {'BYTE','WORD','DWORD','QWORD','XMMWORD','YMMWORD','ZMMWORD','TBYTE','TWORD'}

# ── branch / jump mnemonics (lone operand = code target) ─────────────────────

_BRANCH_OPS = {
    'CALL','JMP',
    'JZ','JNZ','JE','JNE',
    'JA','JB','JL','JG','JLE','JGE','JAE','JBE',
    'JNA','JNB','JNAE','JNBE','JNG','JNL','JNLE','JNGE',
    'JC','JNC','JS','JNS','JO','JNO','JP','JNP','JPE','JPO',
    'JCXZ','JECXZ','JRCXZ',
    'LOOP','LOOPE','LOOPNE','LOOPZ','LOOPNZ',
}

# instructions to drop (no semantic value for embeddings)
_SKIP_INSNS = {'ENDBR64','ENDBR32','NOP','HLT','UD2','DQ','DD','DB','DW'}

# ── memory operand normalizer ─────────────────────────────────────────────────

def _norm_mem(inner):
    """
    Normalize the expression inside [...].
    - register names preserved (uppercased)
    - immediates → 0 / -0
    - operators (+, -, *) preserved
    """
    # split on + or - while keeping delimiters
    parts = re.split(r'([+\-])', inner.strip())
    result = []
    for part in parts:
        p = part.strip()
        if p in ('+', '-', ''):
            result.append(p)
            continue
        if '*' in p:
            sub_parts = p.split('*')
            normed = []
            for s in sub_parts:
                s = s.strip()
                if s.upper() in ALL_REGS:
                    normed.append(s.upper())
                elif _is_imm(s):
                    normed.append(_norm_imm(s))
                else:
                    normed.append(s.upper())
            result.append('*'.join(normed))
        elif p.upper() in ALL_REGS:
            result.append(p.upper())
        elif _is_imm(p):
            result.append(_norm_imm(p))
        else:
            result.append(p.upper())
    return ''.join(result)

# ── token re-joiner ───────────────────────────────────────────────────────────

def _rejoin_tokens(tokens):
    """
    Re-join operand fragments that the corpus split across whitespace.

    Cases handled:
      'qword ptr [RBX + 0x8]'   → 'qword ptr [RBX+0x8]'
      'RAX,qword ptr [0x1234]'  → 'RAX,' + 'qword ptr [0x1234]'
      '[RBX + 0x8]'             → '[RBX+0x8]'
      '[RBX + 0x8],0x1'         → '[RBX+0x8],0x1'
      'EAX,[RBX + 0x8]'         → 'EAX,' + '[RBX+0x8]'
    """
    out = []
    i = 0
    while i < len(tokens):
        t = tokens[i]

        # handle token that has a comma prefix before the size keyword
        # e.g.  "RAX,qword"  →  emit "RAX,"  then re-process "qword ..."
        comma_prefix = ''
        t_body = t
        if ',' in t and not t.startswith('['):
            comma_idx = t.rfind(',')
            maybe_size = t[comma_idx+1:].upper()
            if maybe_size in _PTR_SIZES:
                comma_prefix = t[:comma_idx+1]   # "RAX,"
                t_body = t[comma_idx+1:]          # "qword"
                t = t_body

        # <SIZE> ptr [...]  or  <SIZE> ptr SEG:[...]
        if t.upper() in _PTR_SIZES and i + 1 < len(tokens) and tokens[i+1].lower() == 'ptr':
            size = t.upper()
            i += 2
            # optional segment override token before the bracket: FS:[0x28]
            seg_prefix = ''
            if i < len(tokens):
                seg_m = re.match(r'^([CDEFGS]S):(\[.*)$', tokens[i], re.IGNORECASE)
                if seg_m:
                    seg_prefix = seg_m.group(1).upper() + ':'
                    tokens[i]  = seg_m.group(2)   # rewrite to just '[...'
            if i < len(tokens) and '[' in tokens[i]:
                bracket, i = _collect_bracket(tokens, i)
                merged = f'{size} ptr {seg_prefix}{bracket}'
            else:
                merged = size + ' PTR'
            if comma_prefix:
                out.append(comma_prefix)
            out.append(merged)
            continue

        # open bracket without close → collect until ']'
        if '[' in t and ']' not in t:
            bracket_start = t.index('[')
            prefix = t[:bracket_start]   # e.g. "EAX," from "EAX,[RBX"
            fragment = t[bracket_start:] # starts with '['
            i += 1
            while i < len(tokens) and ']' not in tokens[i]:
                fragment += tokens[i]
                i += 1
            if i < len(tokens):
                fragment += tokens[i]
                i += 1
            if comma_prefix:
                out.append(comma_prefix)
            if prefix:
                out.append(prefix)
            out.append(fragment)
            continue

        if comma_prefix:
            out.append(comma_prefix)
        out.append(t)
        i += 1
    return out


def _collect_bracket(tokens, i):
    """
    Starting at index i (token must start with '['), collect tokens until
    the matching ']' is found.  Returns (full_bracket_string, new_i).
    The returned string includes the outer '[' and ']' and any suffix
    after ']' (e.g. ',RAX').
    """
    fragment = tokens[i]  # starts with '['
    i += 1
    while ']' not in fragment and i < len(tokens):
        fragment += tokens[i]
        i += 1
    return fragment, i

# ── instruction boundary detection ───────────────────────────────────────────

def _is_opcode_token(token):
    """
    True if token looks like a standalone opcode.
    Opcodes are all-alpha, OR alpha-with-digits in known patterns (CVT*2*, FLD1, UD2),
    OR Ghidra-style INSN.PREFIX tokens (e.g. CMPSB.REPE, MOVSQ.REP).
    Must not be a register name.
    """
    t = token.upper()
    if t in ALL_REGS:
        return False
    if re.match(r'^[A-Z]+$', t):
        return True
    # Ghidra REP-prefix format:  INSN.PREFIX
    if re.match(r'^[A-Z]+\.(REP[ENZ]?|REPE|REPNE)$', t):
        return True
    # opcodes containing digits: CVTxx2yy, FLD1, UD2, etc.
    if re.match(r'^[A-Z][A-Z0-9]*[A-Z][A-Z0-9]*$', t) and re.search(r'\d', t):
        if t.startswith('0X'):
            return False
        return True
    return False

# ── main normalizer ───────────────────────────────────────────────────────────

def normalize_line(raw_line):
    """
    Normalize one basic-block sentence.
    Returns a space-separated string of normalized instruction tokens,
    or None if the line is empty / entirely un-normalizable.
    """
    raw_line = raw_line.strip()
    if not raw_line:
        return None

    if ';' in raw_line:
        raw_line = raw_line[:raw_line.index(';')]

    tokens = _rejoin_tokens(raw_line.split())
    instructions = _split_to_instructions(tokens)

    parts = []
    for instr in instructions:
        normed = _norm_instruction(instr)
        if normed:
            parts.append(normed)

    return ' '.join(parts) if parts else None


_PREFIXES = {'LOCK', 'REP', 'REPE', 'REPNE', 'REPNZ', 'REPZ'}

def _split_to_instructions(tokens):
    """
    Group flat token list into per-instruction sub-lists.
    LOCK/REP prefixes are fused with the following opcode token.
    """
    instructions = []
    current = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        t_up = t.upper()
        # fuse LOCK/REP prefix with the next token
        if t_up in _PREFIXES and i + 1 < len(tokens):
            fused = t_up + '~' + tokens[i+1]
            i += 2
            if current:
                instructions.append(current)
            current = [fused]
            continue
        if _is_opcode_token(t) and current:
            instructions.append(current)
            current = [t]
        else:
            if not current:
                current = [t]
            else:
                current.append(t)
        i += 1
    if current:
        instructions.append(current)
    return instructions


def _norm_instruction(tokens):
    """Normalize one instruction → '~'-joined string or None."""
    if not tokens:
        return None

    opcode = tokens[0].upper()

    if opcode in _SKIP_INSNS:
        return None


    if opcode in ('PROC','ENDP','NEAR','FAR','EXTRN'):
        return None

    # Ghidra INSN.PREFIX → PREFIX~INSN
    dot = opcode.find('.')
    if dot > 0:
        insn   = opcode[:dot]
        prefix = opcode[dot+1:]
        opcode = prefix + '~' + insn

    operands = _merge_split_operands(tokens[1:])

    # branch / call with a single operand
    if opcode in _BRANCH_OPS and len(operands) == 1:
        return opcode + '~' + _norm_branch_target(operands[0])

    normed_ops = []
    for op in operands:
        n = _norm_operand(op)
        if n is not None:
            normed_ops.append(n)

    return '~'.join([opcode] + normed_ops)


def _merge_split_operands(operands):
    """
    Re-join operand tokens that _rejoin_tokens had to emit separately because
    a comma preceded a size-keyword or bracket at the token boundary.

    e.g.  ['RAX,', 'qword ptr [0x1234]']       → ['RAX,qword ptr [0x1234]']
          ['RAX,', 'qword ptr FS:[0x28]']       → ['RAX,qword ptr FS:[0x28]']
          ['EAX,', '[RDX+0]']                   → ['EAX,[RDX+0]']
    """
    if not operands:
        return operands
    merged = []
    i = 0
    while i < len(operands):
        cur = operands[i]
        if cur.endswith(',') and i + 1 < len(operands):
            nxt = operands[i+1]
            is_ptr = re.match(
                r'^(BYTE|WORD|DWORD|QWORD|XMMWORD|YMMWORD|ZMMWORD|TBYTE|TWORD)\s+PTR\s+',
                nxt, re.IGNORECASE
            )
            is_bracket = nxt.startswith('[')
            if is_ptr or is_bracket:
                merged.append(cur + nxt)
                i += 2
                continue
        merged.append(cur)
        i += 1
    return merged


def _norm_branch_target(token):
    t = token.upper()
    if t in ALL_REGS:
        return t
    # indirect via memory
    if token.startswith('['):
        inner, suffix = _split_bracket(token)
        return '[' + _norm_mem(inner) + ']'
    # size ptr [...]
    ptr_m = re.match(
        r'^(BYTE|WORD|DWORD|QWORD|XMMWORD|YMMWORD|ZMMWORD|TBYTE|TWORD)\s+PTR\s+(\[.+\])$',
        token, re.IGNORECASE
    )
    if ptr_m:
        inner, _ = _split_bracket(ptr_m.group(2))
        return ptr_m.group(1).upper() + '~PTR~[' + _norm_mem(inner) + ']'
    # plain immediate or symbol → <FOO>
    return '<FOO>'


def _norm_operand(token):
    """Normalize one operand token; return None to drop it."""
    # drop size-only leftovers that weren't merged
    t_up = token.upper()
    if t_up == 'PTR':
        return None
    if t_up in ('OFFSET','SHORT','NEAR','FAR','FLAT:'):
        return None

    # <SIZE> ptr [...]  (already merged by _rejoin_tokens)
    ptr_m = re.match(
        r'^(BYTE|WORD|DWORD|QWORD|XMMWORD|YMMWORD|ZMMWORD|TBYTE|TWORD)\s+PTR\s+(\[.+\])(.*)',
        token, re.IGNORECASE
    )
    if ptr_m:
        size   = ptr_m.group(1).upper()
        bk     = ptr_m.group(2)
        suffix = ptr_m.group(3)          # anything after ']', e.g. ',RAX'
        inner, _ = _split_bracket(bk)
        result = size + '~PTR~[' + _norm_mem(inner) + ']'
        if suffix:
            suf = suffix.lstrip(',')
            result += ',' + _norm_scalar(suf)
        return result

    return _norm_scalar(token)


def _norm_scalar(token):
    """Normalize a token that may be a register, immediate, memory ref, or compound."""
    t_up = token.upper()

    # pure register
    if t_up in ALL_REGS:
        return t_up

    # pure immediate
    if _is_imm(token):
        return _norm_imm(token)

    # <SIZE> ptr [...]  or  <SIZE> ptr SEG:[...]
    ptr_m = re.match(
        r'^(BYTE|WORD|DWORD|QWORD|XMMWORD|YMMWORD|ZMMWORD|TBYTE|TWORD)\s+PTR\s+'
        r'(?:([CDEFGS]S):)?(\[.+\])(.*)',
        token, re.IGNORECASE
    )
    if ptr_m:
        size   = ptr_m.group(1).upper()
        seg    = (ptr_m.group(2) or '').upper()
        bk     = ptr_m.group(3)
        suffix = ptr_m.group(4)
        inner, _ = _split_bracket(bk)
        seg_part = (seg + ':') if seg else ''
        result = size + '~PTR~' + seg_part + '[' + _norm_mem(inner) + ']'
        if suffix:
            suf = suffix.lstrip(',')
            result += ',' + _norm_scalar(suf)
        return result

    # raw [...]  possibly with suffix: [RBX+0x8],RAX  or  [RBX+0x8],0x1
    if token.startswith('['):
        inner, suffix = _split_bracket(token)
        result = '[' + _norm_mem(inner) + ']'
        if suffix:
            suf = suffix.lstrip(',')
            result += ',' + _norm_scalar(suf)
        return result

    # comma-joined pair:  REG,IMM  /  REG,[...]  /  REG,qword ptr [...]
    # find the first comma that isn't inside brackets
    comma = _find_top_comma(token)
    if comma > 0:
        left  = token[:comma]
        right = token[comma+1:]
        return _norm_scalar(left) + ',' + _norm_scalar(right)

    # segment register override:  FS:[...]  DS:0x1234
    seg_m = re.match(r'^([CDEFGS]S):(.+)$', t_up)
    if seg_m:
        seg  = seg_m.group(1)
        rest = token[len(seg)+1:]
        return seg + ':' + _norm_scalar(rest)

    # fallback
    return '<TAG>'


def _find_top_comma(token):
    """Return index of first comma not inside brackets, or -1."""
    depth = 0
    for idx, ch in enumerate(token):
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
        elif ch == ',' and depth == 0:
            return idx
    return -1


def _split_bracket(token):
    """
    Given a token like '[RBX+0x8]' or '[RBX+0x8],RAX',
    return (inner_string, suffix_string).
    inner does NOT include the outer '[' and ']'.
    suffix is everything after ']' (may be empty string).
    """
    assert token.startswith('['), f'expected [ in {token!r}'
    end = token.index(']')
    inner  = token[1:end]
    suffix = token[end+1:]
    return inner, suffix
