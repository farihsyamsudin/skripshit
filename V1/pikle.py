import pickletools

path = "data/maritim.pkl"

with open(path, "rb") as f:
    for opcode, arg, pos in pickletools.genops(f):
        if isinstance(arg, bytes):
            s = arg.decode("latin-1", errors="ignore")
            if "col" in s.lower() or "idx" in s.lower():
                print("Possible string:", s)
