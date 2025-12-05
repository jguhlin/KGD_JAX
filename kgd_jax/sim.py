from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SimulatedData:
    genotypes: np.ndarray  # (n_ind, n_snp), values 0/1/2
    chrom: np.ndarray      # (n_snp,)
    pos: np.ndarray        # (n_snp,)
    sample_ids: np.ndarray # (n_ind,)


def simulate_genotypes(
    n_ind: int,
    n_snp: int,
    seed: Optional[int] = None,
    maf_min: float = 0.05,
    maf_max: float = 0.5,
) -> SimulatedData:
    """Simulate simple SNP genotypes with shared allele frequencies across individuals."""
    rng = np.random.default_rng(seed)

    # Chromosome and position layout: contiguous positions on a single chromosome.
    chrom = np.array(["1"] * n_snp)
    pos = np.arange(1, n_snp + 1, dtype=int)

    # Allele frequencies per SNP.
    p = rng.uniform(maf_min, maf_max, size=n_snp)
    # Genotypes: Binomial(2, p_s) per SNP.
    genotypes = rng.binomial(2, p[None, :], size=(n_ind, n_snp)).astype(np.int8)

    sample_ids = np.array([f"S{i+1}" for i in range(n_ind)], dtype=str)

    return SimulatedData(genotypes=genotypes, chrom=chrom, pos=pos, sample_ids=sample_ids)


def write_chip_csv(sim: SimulatedData, path: Path) -> None:
    """Write genotypes as a 'chip' file compatible with readChip (0/1/2)."""
    n_ind, n_snp = sim.genotypes.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        header = ["ID"] + [f"SNP{s+1}" for s in range(n_snp)]
        fh.write(",".join(header) + "\n")
        for i in range(n_ind):
            row = [sim.sample_ids[i]] + [str(int(g)) for g in sim.genotypes[i, :]]
            fh.write(",".join(row) + "\n")


def write_ra_tab(sim: SimulatedData, depth: int, path: Path) -> None:
    """Write simulated genotypes as an RA .ra.tab file for KGD JAX.

    Uses a fixed depth per SNP and individual:
      - g=0 → (depth,0)
      - g=1 → (depth/2,depth/2)
      - g=2 → (0,depth)
    """
    n_ind, n_snp = sim.genotypes.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        header = ["CHROM", "POS"] + sim.sample_ids.tolist()
        fh.write("\t".join(header) + "\n")
        for s in range(n_snp):
            row = [str(sim.chrom[s]), str(sim.pos[s])]
            for i in range(n_ind):
                g = sim.genotypes[i, s]
                if g == 0:
                    r, a = depth, 0
                elif g == 1:
                    r, a = depth // 2, depth - depth // 2
                elif g == 2:
                    r, a = 0, depth
                else:
                    r, a = 0, 0
                row.append(f"{r},{a}")
            fh.write("\t".join(row) + "\n")

