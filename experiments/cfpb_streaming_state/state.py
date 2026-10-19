"""Reachable allocation accounting, with aliases counted once.

Reports Python object sizes plus owned NumPy buffers, not pickle size or RSS.
Interpreter/module/class code and allocator fragmentation belong to process RSS.
"""
import math
import sys
import types
import numpy as np


def inventory(roots, *, compact=False):
    seen = set()
    entries = []
    groups = {}

    def visit(obj, category, name):
        if obj is None or isinstance(obj, (type, types.ModuleType, types.FunctionType)):
            return
        if id(obj) in seen:
            return
        seen.add(id(obj))
        size = sys.getsizeof(obj)
        payload = obj.nbytes if isinstance(obj, np.ndarray) and obj.base is None else 0
        dtype = str(obj.dtype) if isinstance(obj, np.ndarray) else type(obj).__name__
        row = dict(category=category, name=name, bytes=size, payload_bytes=payload,
                   native_registers=int(obj.size) if payload else None, dtype=dtype)
        if compact:
            group = groups.setdefault((category, dtype), dict(category=category, name=category+'.'+dtype,
                bytes=0, payload_bytes=0, native_registers=0, dtype=dtype, objects=0))
            group['bytes'] += size
            group['payload_bytes'] += payload
            group['native_registers'] += int(obj.size) if payload else 0
            group['objects'] += 1
        else:
            entries.append(row)
        if isinstance(obj, np.ndarray):
            if obj.dtype.hasobject:
                raise TypeError('Object arrays are outside the state contract')
            if obj.base is not None:
                visit(obj.base, category, name + '.base')
        elif isinstance(obj, dict):
            for i, (key, value) in enumerate(obj.items()):
                visit(key, category, name + f'.key{i}')
                visit(value, category, name + f'.value{i}')
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for i, value in enumerate(obj):
                visit(value, category, name + f'[{i}]')
        else:
            if hasattr(obj, '__dict__'):
                visit(obj.__dict__, category, name + '.__dict__')
            for cls in type(obj).__mro__:
                slots = getattr(cls, '__slots__', ())
                if isinstance(slots, str):
                    slots = (slots,)
                for key in slots:
                    if key not in ('__dict__', '__weakref__') and hasattr(obj, key):
                        visit(getattr(obj, key), category, name + '.' + key)

    for category, name, obj in roots:
        visit(obj, category, name)
    return list(groups.values()) if compact else entries


def totals(entries):
    by_category = {}
    for row in entries:
        by_category[row['category']] = by_category.get(row['category'], 0) + row['bytes']
    b = sum(by_category.values())
    return dict(bytes=b, registers64=math.ceil(b / 8),
                payload_bytes=sum(e['payload_bytes'] for e in entries), by_category=by_category)


def deep_bytes(obj):
    return totals(inventory([('workspace', 'object', obj)], compact=True))['bytes']
