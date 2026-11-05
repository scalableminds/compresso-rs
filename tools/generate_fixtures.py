"""Generate reference fixtures with the Python `compresso` package.

Each case stores the stream that `compresso.compress` produces (`<name>.cpso`)
and a manifest entry with the compression parameters plus FNV-1a 64 hashes
of `compresso.decompress` outputs. The input volume equals the full
decompression, so the Rust tests decode each stream, compare the hash, then
re-encode the decoded volume and compare the result byte for byte with the
Python stream.

Usage (from the repository root):

    uv run --no-project --python 3.12 --with compresso==3.3.3 --with numpy==2.3.4 \
        python tools/generate_fixtures.py --output tests/fixtures

    # Larger randomized differential corpus (not checked in):
    uv run ... python tools/generate_fixtures.py --output /tmp/cpso --random 2000 --seed 7
    COMPRESSO_FIXTURES=/tmp/cpso cargo test --release --test fixtures
"""

import argparse
import itertools
import json
import os

import compresso
import numpy as np

DTYPES = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def fnv1a64(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def fortran_bytes(arr: np.ndarray) -> bytes:
    return np.asarray(arr).astype(arr.dtype.newbyteorder("<")).tobytes(order="F")


# ---------------------------------------------------------------- patterns


def splitmix_binary(shape, seed, percent):
    """Voxel i (Fortran order) is 1 iff splitmix64 output i+1 has
    (r >> 32) % 100 < percent. Mirrored in tests/fixtures.rs."""
    n = int(np.prod(shape))
    x = np.uint64(seed) + np.arange(1, n + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    z = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    bits = (z >> np.uint64(32)) % np.uint64(100) < np.uint64(percent)
    return bits.astype(np.uint8).reshape(shape, order="F")


def pattern(kind: str, shape, dtype, rng: np.random.Generator) -> np.ndarray:
    info = np.iinfo(dtype)
    sx, sy, sz = shape
    if kind == "zero":
        return np.zeros(shape, dtype)
    if kind == "max":
        return np.full(shape, info.max, dtype)
    if kind == "random2":
        return rng.integers(0, 2, shape).astype(dtype)
    if kind == "random5":
        return rng.integers(0, 5, shape).astype(dtype)
    if kind == "random_full":
        return rng.integers(0, int(info.max), shape, dtype=dtype, endpoint=True)
    if kind == "high":
        # Labels around the literal escape threshold (max - 7).
        return (np.array(info.max, dtype) - rng.integers(0, 12, shape).astype(dtype)).astype(dtype)
    if kind == "checker":
        x, y, z = np.indices(shape)
        return ((x + y + z) % 2).astype(dtype)
    if kind == "stripes":
        x, y, z = np.indices(shape)
        return ((x // 3 + 2 * (y // 2) + 5 * z) % 7 + 1).astype(dtype)
    if kind == "blobs":
        # Voronoi regions: a realistic, segmentation-like volume.
        n = max(2, int(np.prod(shape) ** 0.5) // 4)
        seeds = rng.uniform(0, 1, (n, 3)) * np.array(shape)
        ids = rng.integers(1, int(min(info.max, 2**40)), n, endpoint=True).astype(dtype)
        grid = np.stack(np.indices(shape), axis=-1).reshape(-1, 1, 3) + 0.5
        nearest = np.argmin(((grid - seeds[None]) ** 2).sum(-1), axis=1)
        out = ids[nearest].reshape(shape)
        out[rng.uniform(0, 1, shape) < 0.01] = 0  # salt
        return out
    if kind == "rings":
        # Concentric labels: components that wrap around others, and
        # components that are split within slices but joined across z.
        x, y, z = np.indices(shape)
        r = np.maximum(abs(x - sx // 2), abs(y - sy // 2))
        out = (r // 2 % 3).astype(dtype)
        out[:, :, ::2] = np.where(out[:, :, ::2] == 1, 4, out[:, :, ::2]).astype(dtype)
        return out
    if kind == "noise_windows":
        # Many distinct boundary windows (exercises the (8,8,1) fallback).
        return rng.integers(0, 2, shape).astype(dtype)
    raise ValueError(kind)


# ------------------------------------------------------------------- cases


class Writer:
    def __init__(self, output):
        self.output = output
        self.cases = []
        os.makedirs(output, exist_ok=True)

    def add(self, name, data, steps, connectivity, random_access, z_ranges=(), extra=None):
        kwargs = {"connectivity": connectivity, "random_access_z_index": random_access}
        if steps is not None:
            kwargs["steps"] = steps
        stream = compresso.compress(data, **kwargs)
        full = compresso.decompress(stream)
        assert full.dtype == data.dtype
        assert np.array_equal(full, data), name
        case = {
            "name": name,
            "width": data.dtype.itemsize,
            "shape": list(data.shape),
            "steps": None if steps is None else list(steps),
            "connectivity": connectivity,
            "random_access_z_index": random_access,
            "hash": f"{fnv1a64(fortran_bytes(full)):016x}",
            "z_ranges": [],
        }
        hdr = compresso.header(stream)
        for start, end in z_ranges:
            part = compresso.decompress(stream, z=(start, end))
            assert np.array_equal(part, data[:, :, start:end])
            case["z_ranges"].append([start, end, f"{fnv1a64(fortran_bytes(part)):016x}"])
        window = hdr["xstep"] * hdr["ystep"] * hdr["zstep"]
        # Python's labels()/remap() misread streams with <= 8 voxel windows
        # (window_dtype returns uint16 for them), so only compare larger windows.
        if window > 8 and data.size > 0:
            labels = compresso.labels(stream)
            assert np.array_equal(labels, np.unique(data))
            case["labels_hash"] = f"{fnv1a64(np.asarray(labels, np.uint64).astype('<u8').tobytes()):016x}"
            mapping = {int(v): int(v) // 2 for v in labels}
            try:
                remapped = compresso.remap(stream, mapping)
                case["remap_half_hash"] = f"{fnv1a64(remapped):016x}"
            except ValueError:
                # Python's raw_windows() also fails when the z index makes
                # the tail length a non-multiple of the window width.
                pass
        if extra:
            case.update(extra)
        with open(os.path.join(self.output, f"{name}.cpso"), "wb") as f:
            f.write(stream)
        self.cases.append(case)

    def finish(self):
        with open(os.path.join(self.output, "cases.json"), "w") as f:
            json.dump(self.cases, f, indent=0)
            f.write("\n")
        print(f"wrote {len(self.cases)} cases to {self.output}")


def z_ranges_for(sz):
    if sz <= 1:
        return []
    return sorted({(0, 1), (sz - 1, sz), (1, sz), (sz // 2, sz // 2 + 1), (0, sz - 1), (0, 0)})


def curated(w: Writer, rng):
    shapes = [(1, 1, 1), (9, 7, 3), (16, 16, 2), (17, 5, 4), (1, 40, 3), (40, 1, 3), (5, 6, 9)]
    kinds = ["zero", "max", "random2", "random5", "random_full", "high", "checker", "stripes", "blobs", "rings"]
    configs = [
        (None, 4, True),
        (None, 4, False),
        (None, 6, False),
        ((2, 2, 1), 4, True),
        ((8, 8, 1), 4, True),
        ((8, 4, 1), 6, False),
        ((4, 4, 2), 4, False),
        ((2, 2, 2), 6, False),
        ((3, 5, 1), 4, True),  # non power of two
        ((4, 4, 4), 4, True),
        ((1, 1, 1), 4, False),
    ]
    for width, kind, shape in itertools.product(DTYPES, kinds, shapes):
        data = pattern(kind, shape, DTYPES[width], rng)
        # Rotate through the configurations to keep the corpus compact.
        steps, conn, ra = configs[len(w.cases) % len(configs)]
        tag = "auto" if steps is None else "x".join(map(str, steps))
        name = f"u{width}_{kind}_{'x'.join(map(str, shape))}_{tag}_c{conn}_{'ra' if ra else 'v0'}"
        w.add(name, data, steps, conn, ra, z_ranges_for(shape[2]) if ra and conn == 4 else ())

    # Every configuration on a realistic volume for each label width.
    for width in DTYPES:
        data = pattern("blobs", (33, 29, 6), DTYPES[width], rng)
        for steps, conn, ra in configs:
            tag = "auto" if steps is None else "x".join(map(str, steps))
            name = f"u{width}_blobs_all_{tag}_c{conn}_{'ra' if ra else 'v0'}"
            w.add(name, data, steps, conn, ra, z_ranges_for(6) if ra and conn == 4 else ())

    # Index width thresholds: 2*sx*sy around 255, 65535.
    for sx, sy in [(127, 1), (128, 1), (1, 127), (11, 11), (181, 181), (182, 181), (128, 256)]:
        data = pattern("blobs", (sx, sy, 3), np.uint16, rng)
        w.add(f"index_width_{sx}x{sy}", data, None, 4, True, z_ranges_for(3))

    # Empty volumes produce a bare header.
    for shape in [(0, 4, 4), (4, 0, 4), (4, 4, 0)]:
        w.add(f"empty_{'x'.join(map(str, shape))}", np.zeros(shape, np.uint32), None, 4, True)
        w.add(f"empty_{'x'.join(map(str, shape))}_c6", np.zeros(shape, np.uint8), (8, 8, 1), 6, True)

    # More than 32767 distinct 4x4x1 windows cannot be run-length encoded in
    # 16 bits, so default steps fall back to (8,8,1) while explicit (4,4,1)
    # raises. The 8M voxel input is generated by the same splitmix64 in Python
    # and Rust; only the hash of the expected stream is stored.
    shape = (4096, 2048, 1)
    data = splitmix_binary(shape, seed=7, percent=35)
    try:
        compresso.compress(data, steps=(4, 4, 1))
        raise AssertionError("expected EncodeError")
    except compresso.EncodeError:
        pass
    stream = compresso.compress(data)
    assert compresso.header(stream)["xstep"] == 8
    assert np.array_equal(compresso.decompress(stream), data)
    w.cases.append({
        "name": "fallback_8x8x1_generated",
        "generator": {"kind": "splitmix_binary", "seed": 7, "percent": 35},
        "width": 1,
        "shape": list(shape),
        "steps": None,
        "connectivity": 4,
        "random_access_z_index": True,
        "stream_len": len(stream),
        "stream_hash": f"{fnv1a64(stream):016x}",
        "explicit_steps_error": [4, 4, 1],
        "z_ranges": [],
    })

    # u8 windows (2x2x2): more than 127 distinct values cannot be encoded.
    data = pattern("noise_windows", (64, 64, 8), np.uint8, rng)
    try:
        compresso.compress(data, steps=(2, 2, 2))
        raise AssertionError("expected EncodeError")
    except compresso.EncodeError:
        pass
    w.add("u8_windows_error_source", data, (8, 8, 1), 4, False, extra={"explicit_steps_error": [2, 2, 2]})

    # Long zero runs that exceed the RLE maximum (127 for 8 bit windows).
    data = np.zeros((256, 256, 2), np.uint8)
    data[255, 255, 1] = 1
    w.add("long_zero_runs_2x2x1", data, (2, 2, 1), 4, True, z_ranges_for(2))
    w.add("long_zero_runs_2x2x2_c6", data, (2, 2, 2), 6, False)

    # A larger realistic volume.
    data = pattern("blobs", (96, 80, 12), np.uint64, rng)
    w.add("u8_blobs_large", data, None, 4, True, z_ranges_for(12))
    w.add("u8_blobs_large_c6", data, None, 6, False)


def randomized(w: Writer, rng, count):
    kinds = ["random2", "random5", "random_full", "high", "checker", "stripes", "blobs", "rings", "zero"]
    for i in range(count):
        width = int(rng.choice(list(DTYPES)))
        shape = tuple(int(v) for v in rng.integers(1, [40, 40, 8], endpoint=True))
        kind = str(rng.choice(kinds))
        conn = int(rng.choice([4, 6]))
        ra = bool(rng.integers(0, 2))
        if rng.integers(0, 3) == 0:
            steps = None
        else:
            while True:
                steps = tuple(int(v) for v in rng.integers(1, [8, 8, 4], endpoint=True))
                if np.prod(steps) <= 64:
                    break
        data = pattern(kind, shape, DTYPES[width], rng)
        try:
            w.add(
                f"rand{i:05d}_u{width}_{kind}",
                data,
                steps,
                conn,
                ra,
                z_ranges_for(shape[2]) if ra and conn == 4 else (),
            )
        except compresso.EncodeError:
            pass  # e.g. too many windows for explicit small steps


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--random", type=int, default=0, help="number of randomized cases instead of the curated corpus")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    w = Writer(args.output)
    if args.random:
        randomized(w, rng, args.random)
    else:
        curated(w, rng)
    w.finish()


if __name__ == "__main__":
    main()
