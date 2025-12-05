from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import grm, io, popgen, ped, plots, merge, tuning, sim, qc


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kgd-jax",
        description="KGD core algorithms in Python/JAX (matrix-free GRM operators).",
    )

    sub = p.add_subparsers(dest="command", required=True)

    # Ingest RA + QC + diagonal G5.
    diag = sub.add_parser(
        "diag",
        help="Compute KGD G5 diagonal (inbreeding) from a .ra.tab file.",
    )
    diag.add_argument("ra_tab", type=Path, help="Input KGD .ra.tab file")
    diag.add_argument(
        "--pmethod",
        choices=["A", "G"],
        default="A",
        help="Allele-frequency estimator: A=reads, G=genotypes (default: A).",
    )
    diag.add_argument(
        "--sampdepth-thresh",
        type=float,
        default=0.01,
        help="Sample mean depth threshold (default: 0.01).",
    )
    diag.add_argument(
        "--snpdepth-thresh",
        type=float,
        default=0.01,
        help="SNP mean depth threshold (default: 0.01).",
    )
    diag.add_argument(
        "--maf-thresh",
        type=float,
        default=1e-9,
        help="Minor allele frequency threshold (default: 1e-9).",
    )
    diag.add_argument(
        "--depth-model",
        choices=["bb", "modp"],
        default="bb",
        help="Depth-to-K model (default: bb).",
    )
    diag.add_argument(
        "--depth-param",
        type=float,
        default=None,
        help="Parameter for depth model (alpha for bb, modp for modp).",
    )
    diag.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV with columns: sample_id, G5_diag.",
    )

    # Small G4 block for a subset of samples.
    block = sub.add_parser(
        "block",
        help=(
            "Compute a small G4 block for a subset of samples, "
            "without forming the full GRM."
        ),
    )
    block.add_argument("ra_tab", type=Path, help="Input KGD .ra.tab file")
    block.add_argument(
        "--samples",
        type=str,
        nargs="+",
        required=True,
        help="Sample IDs to include (subset drawn from RA header).",
    )
    block.add_argument(
        "--pmethod",
        choices=["A", "G"],
        default="A",
        help="Allele-frequency estimator: A=reads, G=genotypes (default: A).",
    )
    block.add_argument(
        "--depth-model",
        choices=["bb", "modp"],
        default="bb",
        help="Depth-to-K model (default: bb).",
    )
    block.add_argument(
        "--depth-param",
        type=float,
        default=None,
        help="Parameter for depth model (alpha for bb, modp for modp).",
    )
    block.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV with G4 block; rows/cols labeled by sample IDs.",
    )

    # Heterozygosity summary.
    het = sub.add_parser(
        "het",
        help="Depth-aware heterozygosity summary (matrix-free, no full GRM).",
    )

    # Hardy–Weinberg diagnostics.
    hw = sub.add_parser(
        "hw",
        help="Hardy–Weinberg disequilibrium and depth-aware LRT statistics.",
    )
    hw.add_argument("ra_tab", type=Path, help="Input KGD .ra.tab file")
    hw.add_argument(
        "--pop-file",
        type=Path,
        required=False,
        help="Optional CSV with columns sample_id,pop giving population for each sample.",
    )
    hw.add_argument(
        "--pmethod",
        choices=["A", "G"],
        default="A",
        help="Allele-frequency estimator for QC (default: A).",
    )
    hw.add_argument(
        "--sampdepth-thresh",
        type=float,
        default=0.01,
        help="Sample mean depth threshold (default: 0.01).",
    )
    hw.add_argument(
        "--snpdepth-thresh",
        type=float,
        default=0.01,
        help="SNP mean depth threshold (default: 0.01).",
    )
    hw.add_argument(
        "--maf-thresh",
        type=float,
        default=1e-9,
        help="Minor allele frequency threshold (default: 1e-9).",
    )
    hw.add_argument(
        "--depth-model",
        choices=["bb", "modp"],
        default="bb",
        help="Depth-to-K model (default: bb).",
    )
    hw.add_argument(
        "--depth-param",
        type=float,
        default=None,
        help="Parameter for depth model (alpha for bb, modp for modp).",
    )
    hw.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "Output CSV with columns: population,chrom,pos,HWdis,l10LRT,x2star,l10pstar,maf,"
            "and optionally l10pstar_pop (per SNP, combined populations)."
        ),
    )
    het.add_argument("ra_tab", type=Path, help="Input KGD .ra.tab file")
    het.add_argument(
        "--pmethod",
        choices=["A", "G"],
        default="A",
        help="Allele-frequency estimator: A=reads, G=genotypes (default: A).",
    )
    het.add_argument(
        "--sampdepth-thresh",
        type=float,
        default=0.01,
        help="Sample mean depth threshold (default: 0.01).",
    )
    het.add_argument(
        "--snpdepth-thresh",
        type=float,
        default=0.01,
        help="SNP mean depth threshold (default: 0.01).",
    )
    het.add_argument(
        "--maf-thresh",
        type=float,
        default=1e-9,
        help="Minor allele frequency threshold (default: 1e-9).",
    )
    het.add_argument(
        "--depth-model",
        choices=["bb", "modp"],
        default="bb",
        help="Depth-to-K model (default: bb).",
    )
    het.add_argument(
        "--depth-param",
        type=float,
        default=None,
        help="Parameter for depth model (alpha for bb, modp for modp).",
    )
    het.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "Output CSV with one row: "
            "neff,ohetstar,ehetstar,ohet,ohet2,ehet."
        ),
    )

    # Fst across populations.
    fst = sub.add_parser(
        "fst",
        help=(
            "Depth-aware Fst (KGD Fst.GBS) using genotype and depth, "
            "matrix-free in GRM."
        ),
    )
    fst.add_argument("ra_tab", type=Path, help="Input KGD .ra.tab file")
    fst.add_argument(
        "--pop-file",
        type=Path,
        required=True,
        help="CSV with columns sample_id,pop giving population for each sample.",
    )
    fst.add_argument(
        "--pmethod",
        choices=["A", "G"],
        default="A",
        help="Allele-frequency estimator for QC (default: A).",
    )
    fst.add_argument(
        "--sampdepth-thresh",
        type=float,
        default=0.01,
        help="Sample mean depth threshold (default: 0.01).",
    )
    fst.add_argument(
        "--snpdepth-thresh",
        type=float,
        default=0.01,
        help="SNP mean depth threshold (default: 0.01).",
    )
    fst.add_argument(
        "--maf-thresh",
        type=float,
        default=1e-9,
        help="Minor allele frequency threshold (default: 1e-9).",
    )
    fst.add_argument(
        "--depth-model",
        choices=["bb", "modp"],
        default="bb",
        help="Depth-to-K model (default: bb).",
    )
    fst.add_argument(
        "--depth-param",
        type=float,
        default=None,
        help="Parameter for depth model (alpha for bb, modp for modp).",
    )
    fst.add_argument(
        "--varadj",
        type=int,
        default=0,
        help="Variance adjustment: 0 for usual Fst, 1 for Weir's correction.",
    )
    fst.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV with per-SNP Fst values and locus metadata.",
    )

    # DAPC on genotype-based PCs (DAPC.GBS analogue).
    dapc = sub.add_parser(
        "dapc",
        help=(
            "Discriminant Analysis of Principal Components based on "
            "genotype PCs (DAPC.GBS analogue)."
        ),
    )
    dapc.add_argument("ra_tab", type=Path, help="Input KGD .ra.tab file")
    dapc.add_argument(
        "--pop-file",
        type=Path,
        required=True,
        help="CSV with columns sample_id,pop giving population for each sample.",
    )
    dapc.add_argument(
        "--pmethod",
        choices=["A", "G"],
        default="A",
        help="Allele-frequency estimator for QC (default: A).",
    )
    dapc.add_argument(
        "--sampdepth-thresh",
        type=float,
        default=0.01,
        help="Sample mean depth threshold (default: 0.01).",
    )
    dapc.add_argument(
        "--snpdepth-thresh",
        type=float,
        default=0.01,
        help="SNP mean depth threshold (default: 0.01).",
    )
    dapc.add_argument(
        "--maf-thresh",
        type=float,
        default=1e-9,
        help="Minor allele frequency threshold (default: 1e-9).",
    )
    dapc.add_argument(
        "--n-pc",
        type=int,
        default=None,
        help="Number of PCs to retain (overrides perc-pc if set).",
    )
    dapc.add_argument(
        "--perc-pc",
        type=float,
        default=90.0,
        help="Percentage of variance to retain in PCs (default: 90).",
    )
    dapc.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "Output CSV with columns: sample_id,pop,PC1..,LD1.. "
            "(PC scores and discriminant coordinates)."
        ),
    )

    # End-to-end pipeline driver.
    run = sub.add_parser(
        "run",
        help=(
            "End-to-end KGD-style pipeline on a .ra.tab file: "
            "QC, G5 diag, inbreeding, heterozygosity, Hardy–Weinberg, "
            "Fst, DAPC, and pedigree verification (if inputs provided)."
        ),
    )
    run.add_argument("ra_tab", type=Path, help="Input KGD .ra.tab file")
    run.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Output file prefix (e.g. 'KGD_run' will create KGD_run.*.csv).",
    )
    run.add_argument(
        "--pop-file",
        type=Path,
        required=False,
        help="Optional CSV with columns sample_id,pop for population-based analyses.",
    )
    run.add_argument(
        "--ped-file",
        type=Path,
        required=False,
        help="Optional pedigree CSV (IndivID,seqID,FatherID,MotherID) for parentage checks.",
    )
    run.add_argument(
        "--pmethod",
        choices=["A", "G"],
        default="A",
        help="Allele-frequency estimator for QC (default: A).",
    )
    run.add_argument(
        "--sampdepth-thresh",
        type=float,
        default=0.01,
        help="Sample mean depth threshold (default: 0.01).",
    )
    run.add_argument(
        "--snpdepth-thresh",
        type=float,
        default=0.01,
        help="SNP mean depth threshold (default: 0.01).",
    )
    run.add_argument(
        "--maf-thresh",
        type=float,
        default=1e-9,
        help="Minor allele frequency threshold (default: 1e-9).",
    )
    run.add_argument(
        "--depth-model",
        choices=["bb", "modp"],
        default="bb",
        help="Depth-to-K model (default: bb).",
    )
    run.add_argument(
        "--depth-param",
        type=float,
        default=None,
        help="Parameter for depth model (alpha for bb, modp for modp).",
    )
    run.add_argument(
        "--rel-thresh",
        type=float,
        default=0.4,
        help="Default relatedness threshold for parent-offspring matches (default: 0.4).",
    )
    run.add_argument(
        "--rel-thresh-f",
        type=float,
        default=None,
        help="Relatedness threshold for fathers (overrides --rel-thresh).",
    )
    run.add_argument(
        "--rel-thresh-m",
        type=float,
        default=None,
        help="Relatedness threshold for mothers (overrides --rel-thresh).",
    )
    run.add_argument(
        "--emm-thresh",
        type=float,
        default=0.01,
        help="Maximum excess mismatch rate for parentage match (default: 0.01).",
    )
    run.add_argument(
        "--mindepth-mm",
        type=float,
        default=1.0,
        help="Minimum depth for mismatch calculations (default: 1).",
    )

    # Merge RA samples by group (technical replicates).
    mrg = sub.add_parser(
        "merge-ra",
        help="Merge RA samples according to a mapping CSV (technical replicates).",
    )
    mrg.add_argument("ra_tab", type=Path, help="Input KGD .ra.tab file.")
    mrg.add_argument(
        "map_csv",
        type=Path,
        help="CSV with columns sample_id,merge_id (or sample,group).",
    )
    mrg.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output merged .ra.tab file.",
    )

    # Depth-model parameter tuning from inbreeding targets.
    fitd = sub.add_parser(
        "fit-depth",
        help=(
            "Fit depth-model parameter (e.g. beta-binomial alpha) to "
            "target inbreeding values, analogous to ssdInb."
        ),
    )
    fitd.add_argument("ra_tab", type=Path, help="Input KGD .ra.tab file.")
    fitd.add_argument(
        "inb_csv",
        type=Path,
        help="CSV with columns sample_id,F_target giving target inbreeding.",
    )
    fitd.add_argument(
        "--dmodel",
        choices=["bb", "modp"],
        default="bb",
        help="Depth model to fit (default: bb).",
    )
    fitd.add_argument(
        "--pmethod",
        choices=["A", "G"],
        default="A",
        help="Allele-frequency estimator for QC (default: A).",
    )
    fitd.add_argument(
        "--sampdepth-thresh",
        type=float,
        default=0.01,
        help="Sample mean depth threshold (default: 0.01).",
    )
    fitd.add_argument(
        "--snpdepth-thresh",
        type=float,
        default=0.01,
        help="SNP mean depth threshold (default: 0.01).",
    )
    fitd.add_argument(
        "--maf-thresh",
        type=float,
        default=1e-9,
        help="Minor allele frequency threshold (default: 1e-9).",
    )
    fitd.add_argument(
        "--bounds",
        type=float,
        nargs=2,
        default=None,
        help="Parameter bounds [low high]; default depends on model.",
    )
    fitd.add_argument(
        "--tol",
        type=float,
        default=0.05,
        help="Tolerance on parameter interval (default: 0.05).",
    )
    fitd.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV with fitted parameter and sum of squares.",
    )

    # Plotting helpers using matplotlib.
    pdapc = sub.add_parser(
        "plot-dapc",
        help="DAPC scatter plot from a dapc CSV produced by dapc/run.",
    )
    pdapc.add_argument("dapc_csv", type=Path, help="DAPC CSV (prefix.dapc.csv).")
    pdapc.add_argument(
        "--x-axis",
        type=str,
        default="LD1",
        help="Column for x-axis (default: LD1).",
    )
    pdapc.add_argument(
        "--y-axis",
        type=str,
        default="LD2",
        help="Column for y-axis (default: LD2).",
    )
    pdapc.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output PNG path for DAPC scatter.",
    )

    pfst = sub.add_parser(
        "plot-fst",
        help="Manhattan-style plot from an Fst CSV produced by fst/run.",
    )
    pfst.add_argument("fst_csv", type=Path, help="Fst CSV (prefix.fst.csv).")
    pfst.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output PNG path for Fst Manhattan plot.",
    )

    phw = sub.add_parser(
        "plot-hw",
        help="HW disequilibrium vs MAF scatter from an HW CSV produced by hw/run.",
    )
    phw.add_argument("hw_csv", type=Path, help="HW CSV (prefix.hw.csv).")
    phw.add_argument(
        "--population",
        type=str,
        default=None,
        help="Optional population label to filter on.",
    )
    phw.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output PNG path for HW vs MAF scatter.",
    )

    # SNP selection for LD / Ne analysis.
    snpsel = sub.add_parser(
        "snpselect",
        help="Select SNP pairs for LD/Ne analysis (snpselection analogue).",
    )
    snpsel.add_argument(
        "pos_csv",
        type=Path,
        help="CSV with columns chrom,pos describing SNP coordinates in order.",
    )
    snpsel.add_argument(
        "--nsnp-per-chrom",
        type=int,
        default=100,
        help="Number of SNPs to select per chromosome (default 100).",
    )
    snpsel.add_argument(
        "--seltype",
        type=str,
        default="centre",
        help="Selection type: 'centre', 'even', or 'random' (default 'centre').",
    )
    snpsel.add_argument(
        "--chrom-use",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of chromosome labels to include (default: all).",
    )
    snpsel.add_argument(
        "--randseed",
        type=int,
        default=None,
        help="Random seed for seltype='random'.",
    )
    snpsel.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV with columns snp1,snp2 (0-based indices).",
    )

    # Data simulation for cross-checks with original KGD.
    simcmd = sub.add_parser(
        "simulate",
        help="Simulate genotypes and write chip CSV + RA .ra.tab for R and JAX KGD.",
    )
    simcmd.add_argument(
        "--n-ind",
        type=int,
        default=200,
        help="Number of individuals to simulate (default 200).",
    )
    simcmd.add_argument(
        "--n-snp",
        type=int,
        default=2000,
        help="Number of SNPs to simulate (default 2000).",
    )
    simcmd.add_argument(
        "--depth",
        type=int,
        default=20,
        help="Fixed read depth per individual/SNP in RA output (default 20).",
    )
    simcmd.add_argument(
        "--prefix",
        type=str,
        default="sim",
        help="Prefix for output files (default 'sim').",
    )
    simcmd.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for simulation (default 123).",
    )

    # VCF -> RA-store converter using original vcf2ra.py + RA->Zarr.
    v2k = sub.add_parser(
        "vcf2kgd",
        help=(
            "Convert a VCF with depth fields to a compressed KGD-JAX store "
            "using orig_kgd/vcf2ra.py followed by RA->Zarr."
        ),
    )
    v2k.add_argument("vcf", type=Path, help="Input VCF file.")
    v2k.add_argument(
        "--store",
        type=Path,
        required=True,
        help="Output Zarr store path for KGD-JAX (e.g. data.kgd.zarr).",
    )
    v2k.add_argument(
        "--keep-ra",
        action="store_true",
        help="Keep the intermediate .ra.tab produced by vcf2ra.py (default: delete).",
    )

    # Best-parent search across all candidates (simplified bestmatch analogue).
    pedbest = sub.add_parser(
        "ped-best",
        help=(
            "Search for best and second-best candidate parents per offspring "
            "using G4 relatedness and depth-aware EMM."
        ),
    )
    pedbest.add_argument("ra_tab", type=Path, help="Input KGD .ra.tab file.")
    pedbest.add_argument(
        "ped_file",
        type=Path,
        help="Pedigree CSV with at least IndivID,seqID.",
    )
    pedbest.add_argument(
        "--pmethod",
        choices=["A", "G"],
        default="A",
        help="Allele-frequency estimator for QC (default: A).",
    )
    pedbest.add_argument(
        "--sampdepth-thresh",
        type=float,
        default=0.01,
        help="Sample mean depth threshold (default: 0.01).",
    )
    pedbest.add_argument(
        "--snpdepth-thresh",
        type=float,
        default=0.01,
        help="SNP mean depth threshold (default: 0.01).",
    )
    pedbest.add_argument(
        "--maf-thresh",
        type=float,
        default=1e-9,
        help="Minor allele frequency threshold (default: 1e-9).",
    )
    pedbest.add_argument(
        "--depth-model",
        choices=["bb", "modp"],
        default="bb",
        help="Depth-to-K model (default: bb).",
    )
    pedbest.add_argument(
        "--depth-param",
        type=float,
        default=None,
        help="Parameter for depth model (alpha for bb, modp for modp).",
    )
    pedbest.add_argument(
        "--criterion",
        choices=["rel", "EMM"],
        default="rel",
        help="Best-parent criterion: rel (relatedness) or EMM (excess mismatch).",
    )
    pedbest.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV with best/second-best candidate parents per offspring.",
    )

    # Parentage / pedigree verification.
    pedp = sub.add_parser(
        "ped",
        help=(
            "Depth-aware parentage verification using G4 relatedness and "
            "mismatch expectations for recorded parents."
        ),
    )
    pedp.add_argument("ra_tab", type=Path, help="Input KGD .ra.tab file")
    pedp.add_argument("ped_file", type=Path, help="Pedigree CSV (IndivID,seqID, FatherID,MotherID).")
    pedp.add_argument(
        "--pmethod",
        choices=["A", "G"],
        default="A",
        help="Allele-frequency estimator for QC (default: A).",
    )
    pedp.add_argument(
        "--sampdepth-thresh",
        type=float,
        default=0.01,
        help="Sample mean depth threshold (default: 0.01).",
    )
    pedp.add_argument(
        "--snpdepth-thresh",
        type=float,
        default=0.01,
        help="SNP mean depth threshold (default: 0.01).",
    )
    pedp.add_argument(
        "--maf-thresh",
        type=float,
        default=1e-9,
        help="Minor allele frequency threshold (default: 1e-9).",
    )
    pedp.add_argument(
        "--depth-model",
        choices=["bb", "modp"],
        default="bb",
        help="Depth-to-K model (default: bb).",
    )
    pedp.add_argument(
        "--depth-param",
        type=float,
        default=None,
        help="Parameter for depth model (alpha for bb, modp for modp).",
    )
    pedp.add_argument(
        "--rel-thresh",
        type=float,
        default=0.4,
        help="Default relatedness threshold for parent-offspring matches (default: 0.4).",
    )
    pedp.add_argument(
        "--rel-thresh-f",
        type=float,
        default=None,
        help="Relatedness threshold for fathers (overrides --rel-thresh).",
    )
    pedp.add_argument(
        "--rel-thresh-m",
        type=float,
        default=None,
        help="Relatedness threshold for mothers (overrides --rel-thresh).",
    )
    pedp.add_argument(
        "--emm-thresh",
        type=float,
        default=0.01,
        help="Maximum excess mismatch rate for a match (default: 0.01).",
    )
    pedp.add_argument(
        "--mindepth-mm",
        type=float,
        default=1.0,
        help="Minimum depth for both individuals for mismatch calculations (default: 1).",
    )
    pedp.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV with pedigree plus columns: Inb, FatherRel/EMM/Match/Inb, MotherRel/EMM/Match/Inb.",
    )

    # Inbreeding coefficients F_i ≈ G5_diag_i - 1.
    inb = sub.add_parser(
        "inbreed",
        help="Compute depth-corrected inbreeding coefficients from a .ra.tab file.",
    )
    inb.add_argument("ra_tab", type=Path, help="Input KGD .ra.tab file")
    inb.add_argument(
        "--pmethod",
        choices=["A", "G"],
        default="A",
        help="Allele-frequency estimator: A=reads, G=genotypes (default: A).",
    )
    inb.add_argument(
        "--sampdepth-thresh",
        type=float,
        default=0.01,
        help="Sample mean depth threshold (default: 0.01).",
    )
    inb.add_argument(
        "--snpdepth-thresh",
        type=float,
        default=0.01,
        help="SNP mean depth threshold (default: 0.01).",
    )
    inb.add_argument(
        "--maf-thresh",
        type=float,
        default=1e-9,
        help="Minor allele frequency threshold (default: 1e-9).",
    )
    inb.add_argument(
        "--depth-model",
        choices=["bb", "modp"],
        default="bb",
        help="Depth-to-K model (default: bb).",
    )
    inb.add_argument(
        "--depth-param",
        type=float,
        default=None,
        help="Parameter for depth model (alpha for bb, modp for modp).",
    )
    inb.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV with columns: sample_id, F_hat (G5_diag-1).",
    )

    return p


def cmd_diag(args: argparse.Namespace) -> None:
    ra = io.read_ra_tab(args.ra_tab)
    qc_result = qc.run_qc(
        ra,
        sampdepth_thresh=args.sampdepth_thresh,
        snpdepth_thresh=args.snpdepth_thresh,
        maf_thresh=args.maf_thresh,
        pmethod=args.pmethod,
    )

    op = grm.build_grm_operator(
        depth=qc_result.depth,
        genon=qc_result.genon,
        p=qc_result.p,
        dmodel=args.depth_model,
        dparam=args.depth_param,
    )
    G5d = op.diag_G5()

    kept_ids = np.array(ra.sample_ids)[qc_result.keep_ind]
    out_df = pd.DataFrame(
        {"sample_id": kept_ids, "G5_diag": np.asarray(G5d, dtype=np.float64)}
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)


def cmd_block(args: argparse.Namespace) -> None:
    ra = io.read_ra_tab(args.ra_tab)

    # Map requested sample IDs to indices.
    sample_to_idx = {sid: i for i, sid in enumerate(ra.sample_ids)}
    try:
        idx = [sample_to_idx[s] for s in args.samples]
    except KeyError as e:
        missing = str(e.args[0])
        raise SystemExit(f"Sample ID '{missing}' not found in RA header") from e

    # Run QC globally, then restrict to the requested subset.
    qc_result = qc.run_qc(ra, pmethod=args.pmethod)
    op = grm.build_grm_operator(
        depth=qc_result.depth,
        genon=qc_result.genon,
        p=qc_result.p,
        dmodel=args.depth_model,
        dparam=args.depth_param,
    )

    # Translate original indices to post-QC indices.
    kept_indices = np.where(qc_result.keep_ind)[0]
    idx_post = []
    for i in idx:
        pos = np.where(kept_indices == i)[0]
        if len(pos) == 0:
            raise SystemExit(
                f"Sample '{ra.sample_ids[i]}' was removed during QC; "
                "relax thresholds or choose another sample."
            )
        idx_post.append(int(pos[0]))

    G4_block = op.submatrix_G4(ind_i=idx_post)
    block_np = np.asarray(G4_block, dtype=np.float64)
    labels = [ra.sample_ids[i] for i in np.array(kept_indices)[idx_post]]

    out_df = pd.DataFrame(block_np, index=labels, columns=labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out)


def cmd_het(args: argparse.Namespace) -> None:
    ra = io.read_ra_tab(args.ra_tab)
    qc_result = qc.run_qc(
        ra,
        sampdepth_thresh=args.sampdepth_thresh,
        snpdepth_thresh=args.snpdepth_thresh,
        maf_thresh=args.maf_thresh,
        pmethod=args.pmethod,
    )

    depth_np = np.asarray(qc_result.depth)
    genon_np = np.asarray(qc_result.genon)
    p_np = np.asarray(qc_result.p)
    depth2K_fn = grm.make_depth2K(dmodel=args.depth_model, param=args.depth_param)

    het_res = popgen.heterozygosity(
        depth=depth_np,
        genon=genon_np,
        p=p_np,
        depth2K=depth2K_fn,
    )

    out_df = pd.DataFrame(
        [
            {
                "neff": het_res.neff,
                "ohetstar": het_res.ohetstar,
                "ehetstar": het_res.ehetstar,
                "ohet": het_res.ohet,
                "ohet2": het_res.ohet2,
                "ehet": het_res.ehet,
            }
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)


def _load_populations(pop_file: Path, sample_ids: Sequence[str]) -> np.ndarray:
    df = pd.read_csv(pop_file)
    cols = {c.lower(): c for c in df.columns}
    if "sample_id" in cols:
        sample_col = cols["sample_id"]
    elif "sample" in cols:
        sample_col = cols["sample"]
    else:
        raise SystemExit("Population file must have a 'sample_id' or 'sample' column.")

    if "pop" in cols:
        pop_col = cols["pop"]
    elif "population" in cols:
        pop_col = cols["population"]
    else:
        raise SystemExit("Population file must have a 'pop' or 'population' column.")

    pop_map = dict(zip(df[sample_col].astype(str), df[pop_col].astype(str)))
    pops = []
    missing = []
    for sid in sample_ids:
        if sid in pop_map:
            pops.append(pop_map[sid])
        else:
            missing.append(sid)
    if missing:
        missing_str = ", ".join(missing[:5])
        raise SystemExit(
            f"Population assignments missing for {len(missing)} samples, "
            f"e.g. {missing_str}."
        )
    return np.array(pops, dtype=str)


def cmd_fst(args: argparse.Namespace) -> None:
    ra = io.read_ra_tab(args.ra_tab)
    qc_result = qc.run_qc(
        ra,
        sampdepth_thresh=args.sampdepth_thresh,
        snpdepth_thresh=args.snpdepth_thresh,
        maf_thresh=args.maf_thresh,
        pmethod=args.pmethod,
    )

    kept_ids = np.array(ra.sample_ids)[qc_result.keep_ind]
    pops_full = _load_populations(args.pop_file, ra.sample_ids)
    pops_qc = pops_full[qc_result.keep_ind]

    depth_np = np.asarray(qc_result.depth)
    genon_np = np.asarray(qc_result.genon)
    depth2K_fn = grm.make_depth2K(dmodel=args.depth_model, param=args.depth_param)

    fst_res = popgen.fst_gbs(
        depth=depth_np,
        genon=genon_np,
        populations=pops_qc,
        depth2K=depth2K_fn,
        varadj=args.varadj,
    )

    snp_chrom = ra.chrom[qc_result.keep_snp]
    snp_pos = ra.pos[qc_result.keep_snp]
    fst_vals = fst_res.fst

    out_df = pd.DataFrame(
        {
            "chrom": snp_chrom,
            "pos": snp_pos,
            "Fst": fst_vals,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    # Print a brief summary to stdout.
    mean_fst = np.nanmean(fst_vals)
    median_fst = np.nanmedian(fst_vals)
    print(f"Fst mean: {mean_fst:.4f}  median: {median_fst:.4f}")


def cmd_hw(args: argparse.Namespace) -> None:
    ra = io.read_ra_tab(args.ra_tab)
    qc_result = qc.run_qc(
        ra,
        sampdepth_thresh=args.sampdepth_thresh,
        snpdepth_thresh=args.snpdepth_thresh,
        maf_thresh=args.maf_thresh,
        pmethod=args.pmethod,
    )

    depth_np = np.asarray(qc_result.depth)
    genon_np = np.asarray(qc_result.genon)
    depth2K_fn = grm.make_depth2K(dmodel=args.depth_model, param=args.depth_param)

    if args.pop_file is not None:
        pops_full = _load_populations(args.pop_file, ra.sample_ids)
        pops_qc = pops_full[qc_result.keep_ind]
    else:
        pops_qc = None

    hw_res = popgen.hw_pops(
        genon=genon_np,
        depth=depth_np,
        populations=pops_qc,
        depth2K=depth2K_fn,
    )

    chrom = ra.chrom[qc_result.keep_snp]
    pos = ra.pos[qc_result.keep_snp]
    n_snp = chrom.shape[0]

    rows = []
    for ipop, pop in enumerate(hw_res.popnames):
        for s in range(n_snp):
            rows.append(
                {
                    "population": pop,
                    "chrom": chrom[s],
                    "pos": pos[s],
                    "HWdis": hw_res.HWdis[ipop, s],
                    "l10LRT": hw_res.l10LRT[ipop, s],
                    "x2star": hw_res.x2star[ipop, s],
                    "l10pstar": hw_res.l10pstar[ipop, s],
                    "maf": hw_res.maf[ipop, s],
                    "l10pstar_pop": (
                        hw_res.l10pstar_pop[s]
                        if hw_res.l10pstar_pop is not None
                        else np.nan
                    ),
                }
            )

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)


def cmd_ped(args: argparse.Namespace) -> None:
    ra = io.read_ra_tab(args.ra_tab)
    qc_result = qc.run_qc(
        ra,
        sampdepth_thresh=args.sampdepth_thresh,
        snpdepth_thresh=args.snpdepth_thresh,
        maf_thresh=args.maf_thresh,
        pmethod=args.pmethod,
    )

    op = grm.build_grm_operator(
        depth=qc_result.depth,
        genon=qc_result.genon,
        p=qc_result.p,
        dmodel=args.depth_model,
        dparam=args.depth_param,
    )
    G5d = op.diag_G5()

    depth_np = np.asarray(qc_result.depth)
    genon_np = np.asarray(qc_result.genon)
    p_np = np.asarray(qc_result.p)
    depth2K_fn = grm.make_depth2K(dmodel=args.depth_model, param=args.depth_param)

    ped_df = pd.read_csv(args.ped_file)

    relF = args.rel_thresh_f if args.rel_thresh_f is not None else args.rel_thresh
    relM = args.rel_thresh_m if args.rel_thresh_m is not None else args.rel_thresh

    ped_out = ped.check_parents(
        ped_df=ped_df,
        sample_ids=ra.sample_ids,
        keep_ind_mask=qc_result.keep_ind,
        G5_diag=np.asarray(G5d),
        grm_op=op,
        depth_np=depth_np,
        genon_np=genon_np,
        p_np=p_np,
        depth2K=depth2K_fn,
        rel_threshF=relF,
        rel_threshM=relM,
        emm_thresh=args.emm_thresh,
        mindepth_mm=args.mindepth_mm,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ped_out.to_csv(args.out, index=False)


def cmd_dapc(args: argparse.Namespace) -> None:
    ra = io.read_ra_tab(args.ra_tab)
    qc_result = qc.run_qc(
        ra,
        sampdepth_thresh=args.sampdepth_thresh,
        snpdepth_thresh=args.snpdepth_thresh,
        maf_thresh=args.maf_thresh,
        pmethod=args.pmethod,
    )

    pops_full = _load_populations(args.pop_file, ra.sample_ids)
    pops_qc = pops_full[qc_result.keep_ind]

    genon_np = np.asarray(qc_result.genon)
    p_np = np.asarray(qc_result.p)
    kept_ids = np.array(ra.sample_ids)[qc_result.keep_ind]

    dapc_res = popgen.dapc_from_genotypes(
        genon=genon_np,
        p=p_np,
        populations=pops_qc,
        sample_ids=kept_ids,
        n_pca=args.n_pc,
        perc_pca=args.perc_pc,
    )

    n_pc = dapc_res.pc_scores.shape[1]
    n_disc = dapc_res.lda_scores.shape[1]

    data = {
        "sample_id": dapc_res.sample_ids,
        "pop": dapc_res.populations,
    }
    for i in range(n_pc):
        data[f"PC{i+1}"] = dapc_res.pc_scores[:, i]
    for j in range(n_disc):
        data[f"LD{j+1}"] = dapc_res.lda_scores[:, j]

    out_df = pd.DataFrame(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)


def cmd_merge_ra(args: argparse.Namespace) -> None:
    ra = io.read_ra_tab(args.ra_tab)
    mapping = merge.load_merge_mapping(args.map_csv)
    merged = merge.merge_ra_samples(ra, mapping)

    # Write merged RA in the same format as vcf2ra output.
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        header = ["CHROM", "POS"] + merged.sample_ids
        fh.write("\t".join(header) + "\n")
        for idx in range(merged.n_snp):
            row = [str(merged.chrom[idx]), str(merged.pos[idx])]
            for j in range(merged.n_ind):
                row.append(f"{int(merged.ref[j, idx])},{int(merged.alt[j, idx])}")
            fh.write("\t".join(row) + "\n")


def cmd_fit_depth(args: argparse.Namespace) -> None:
    ra = io.read_ra_tab(args.ra_tab)
    qc_result = qc.run_qc(
        ra,
        sampdepth_thresh=args.sampdepth_thresh,
        snpdepth_thresh=args.snpdepth_thresh,
        maf_thresh=args.maf_thresh,
        pmethod=args.pmethod,
    )

    # Load target inbreeding and align to QC-kept samples.
    inb_df = pd.read_csv(args.inb_csv)
    cols = {c.lower(): c for c in inb_df.columns}
    if "sample_id" in cols:
        sample_col = cols["sample_id"]
    elif "sample" in cols:
        sample_col = cols["sample"]
    else:
        raise SystemExit("Inbreeding CSV must have 'sample_id' or 'sample' column.")
    if "f_target" in cols:
        f_col = cols["f_target"]
    elif "f_hat" in cols:
        f_col = cols["f_hat"]
    elif "fhat" in cols:
        f_col = cols["fhat"]
    elif "f" in cols:
        f_col = cols["f"]
    else:
        raise SystemExit("Inbreeding CSV must have an F column (F_target/Fhat/F).")

    target_map = dict(
        zip(
            inb_df[sample_col].astype(str),
            inb_df[f_col].astype(float),
        )
    )
    kept_ids = np.array(ra.sample_ids)[qc_result.keep_ind]
    ind_indices = []
    inb_target = []
    for i, sid in enumerate(kept_ids):
        if sid in target_map:
            ind_indices.append(i)
            inb_target.append(target_map[sid])
    if not ind_indices:
        raise SystemExit("No overlap between QC-kept samples and inbreeding targets.")

    ind_indices_np = np.asarray(ind_indices, dtype=np.int32)
    inb_target_np = np.asarray(inb_target, dtype=np.float64)

    depth = qc_result.depth
    genon = qc_result.genon
    p = qc_result.p

    if args.bounds is None:
        if args.dmodel == "bb":
            bounds = (0.1, 200.0)
        else:
            bounds = (0.51, 0.999)
    else:
        low, high = args.bounds
        bounds = (low, high)

    fit_res = tuning.fit_depth_param_inb(
        depth=depth,
        genon=genon,
        p=p,
        inb_target=inb_target_np,
        dmodel=args.dmodel,
        ind_indices=ind_indices_np,
        snp_indices=None,
        bounds=bounds,
        tol=args.tol,
    )

    out_df = pd.DataFrame(
        [
            {
                "dmodel": fit_res.dmodel,
                "param_opt": fit_res.param_opt,
                "ss_min": fit_res.ss_min,
                "n_ind": fit_res.n_ind,
                "n_snp": fit_res.n_snp,
            }
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)


def cmd_simulate(args: argparse.Namespace) -> None:
    """Simulate genotypes and write chip + RA + JAX-native store."""
    prefix = Path(args.prefix)
    simdata = sim.simulate_genotypes(
        n_ind=args.n_ind,
        n_snp=args.n_snp,
        seed=args.seed,
    )

    chip_path = prefix.with_suffix(".chip.csv")
    ra_path = prefix.with_suffix(".ra.tab")

    sim.write_chip_csv(simdata, chip_path)
    sim.write_ra_tab(simdata, depth=args.depth, path=ra_path)

    print(f"Wrote chip file for R KGD: {chip_path}")
    print(f"Wrote RA file for KGD JAX: {ra_path}")

    # Also write a compressed Zarr store for JAX-native workflows.
    try:
        ra = io.read_ra_tab(ra_path)
        store_path = prefix.with_suffix(".kgd.zarr")
        io.write_ra_store(ra, store_path)
        print(f"Wrote KGD-JAX Zarr store: {store_path}")
    except Exception as exc:  # pragma: no cover - best-effort convenience
        print(f"Warning: failed to write KGD-JAX store from RA: {exc}")


def cmd_ped_best(args: argparse.Namespace) -> None:
    """Search for best and second-best candidate parents per offspring."""
    ra = io.read_ra_tab(args.ra_tab)
    qc_result = qc.run_qc(
        ra,
        sampdepth_thresh=args.sampdepth_thresh,
        snpdepth_thresh=args.snpdepth_thresh,
        maf_thresh=args.maf_thresh,
        pmethod=args.pmethod,
    )

    depth_np = np.asarray(qc_result.depth)
    genon_np = np.asarray(qc_result.genon)
    p_np = np.asarray(qc_result.p)
    depth2K_fn = grm.make_depth2K(dmodel=args.depth_model, param=args.depth_param)

    op = grm.build_grm_operator(
        depth=qc_result.depth,
        genon=qc_result.genon,
        p=qc_result.p,
        dmodel=args.depth_model,
        dparam=args.depth_param,
    )

    ped_df = pd.read_csv(args.ped_file)
    # Map seqID in ped to QC indices.
    name_to_ra = {sid: i for i, sid in enumerate(ra.sample_ids)}
    ra_keep_idx = np.where(qc_result.keep_ind)[0]
    ra_to_qc = {int(ra_idx): pos for pos, ra_idx in enumerate(ra_keep_idx)}

    offspring_idx = []
    for _, row in ped_df.iterrows():
        seq = str(row["seqID"])
        if seq in name_to_ra:
            ra_i = name_to_ra[seq]
            if ra_i in ra_to_qc:
                offspring_idx.append(ra_to_qc[ra_i])
    offspring_idx = np.asarray(offspring_idx, dtype=int)

    # Candidate parents: all QC-kept individuals.
    parent_idx = np.arange(depth_np.shape[0], dtype=int)

    if args.criterion == "rel":
        best = ped.best_parents_by_relatedness(
            grm_op=op,
            offspring_idx=offspring_idx,
            parent_idx=parent_idx,
        )
        # Compute EMM only for best parents for context.
        EMM_best = []
        for oi, pi in zip(best.offspring_idx, best.best_parent_idx):
            mm = ped.mismatch_par_pair(
                depth=depth_np,
                genon=genon_np,
                p=p_np,
                depth2K=depth2K_fn,
                i=int(oi),
                j=int(pi),
            )
            EMM_best.append(mm.mmrate - mm.exp_mmrate)
        EMM_best = np.asarray(EMM_best, dtype=float)
    else:
        best = ped.best_parents_by_emm(
            depth=depth_np,
            genon=genon_np,
            p=p_np,
            depth2K=depth2K_fn,
            offspring_idx=offspring_idx,
            parent_idx=parent_idx,
        )
        # Fill relatedness for best and second-best from GRMOperator.
        G_block = np.asarray(
            op.submatrix_G4(best.offspring_idx, best.best_parent_idx), dtype=float
        )
        rel_best = G_block[:, 0] if G_block.ndim == 2 and G_block.shape[1] == 1 else np.diag(G_block)
        G_block2 = np.asarray(
            op.submatrix_G4(best.offspring_idx, best.second_parent_idx), dtype=float
        )
        rel_second = (
            G_block2[:, 0] if G_block2.ndim == 2 and G_block2.shape[1] == 1 else np.diag(G_block2)
        )
        best.rel_best = rel_best
        best.rel_second = rel_second
        EMM_best = np.full_like(best.rel_best, np.nan)

    # Map indices back to sample IDs.
    kept_ids = np.array(ra.sample_ids)[qc_result.keep_ind]
    off_ids = kept_ids[best.offspring_idx]
    par1_ids = kept_ids[best.best_parent_idx]
    par2_ids = kept_ids[best.second_parent_idx]

    out_df = pd.DataFrame(
        {
            "OffspringSeqID": off_ids,
            "BestParentSeqID": par1_ids,
            "SecondParentSeqID": par2_ids,
            "RelBest": best.rel_best,
            "RelSecond": best.rel_second,
            "RelParents": best.rel_parents,
            "EMM_Best": EMM_best,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)


def cmd_vcf2kgd(args: argparse.Namespace) -> None:
    """Convert a VCF to a compressed KGD-JAX store via orig_kgd/vcf2ra.py."""
    import subprocess
    import sys

    vcf_path = args.vcf
    store_path = args.store

    # Step 1: call the original vcf2ra.py to get a .ra.tab file.
    vcf2ra_script = Path("orig_kgd") / "vcf2ra.py"
    if not vcf2ra_script.exists():
        raise SystemExit(f"vcf2ra.py not found at {vcf2ra_script}")

    cmd = [sys.executable, str(vcf2ra_script), str(vcf_path)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    ra_path = Path(str(vcf_path) + ".ra.tab")
    if not ra_path.exists():
        raise SystemExit(f"Expected RA file not found: {ra_path}")

    # Step 2: convert RA -> Zarr store.
    ra = io.read_ra_tab(ra_path)
    io.write_ra_store(ra, store_path)
    print(f"Wrote KGD-JAX Zarr store: {store_path}")

    # Optionally remove the intermediate RA.
    if not args.keep_ra:
        try:
            ra_path.unlink()
            print(f"Removed intermediate RA file: {ra_path}")
        except OSError as exc:  # pragma: no cover
            print(f"Warning: failed to remove RA file {ra_path}: {exc}")


def cmd_run(args: argparse.Namespace) -> None:
    """End-to-end pipeline driver combining core analyses."""
    ra = io.read_ra_tab(args.ra_tab)
    qc_result = qc.run_qc(
        ra,
        sampdepth_thresh=args.sampdepth_thresh,
        snpdepth_thresh=args.snpdepth_thresh,
        maf_thresh=args.maf_thresh,
        pmethod=args.pmethod,
    )

    prefix = args.prefix
    base = Path(prefix)
    base.parent.mkdir(parents=True, exist_ok=True)

    # Core GRM operator and diag.
    op = grm.build_grm_operator(
        depth=qc_result.depth,
        genon=qc_result.genon,
        p=qc_result.p,
        dmodel=args.depth_model,
        dparam=args.depth_param,
    )
    G5d = op.diag_G5()
    F_hat = G5d - 1.0

    kept_ids = np.array(ra.sample_ids)[qc_result.keep_ind]

    # G5 diagonal.
    diag_df = pd.DataFrame(
        {"sample_id": kept_ids, "G5_diag": np.asarray(G5d, dtype=np.float64)}
    )
    diag_df.to_csv(f"{prefix}.G5diag.csv", index=False)

    # Inbreeding.
    inb_df = pd.DataFrame(
        {"sample_id": kept_ids, "F_hat": np.asarray(F_hat, dtype=np.float64)}
    )
    inb_df.to_csv(f"{prefix}.inbreed.csv", index=False)

    # Heterozygosity.
    depth_np = np.asarray(qc_result.depth)
    genon_np = np.asarray(qc_result.genon)
    p_np = np.asarray(qc_result.p)
    depth2K_fn = grm.make_depth2K(dmodel=args.depth_model, param=args.depth_param)

    het_res = popgen.heterozygosity(
        depth=depth_np,
        genon=genon_np,
        p=p_np,
        depth2K=depth2K_fn,
    )
    het_df = pd.DataFrame(
        [
            {
                "neff": het_res.neff,
                "ohetstar": het_res.ohetstar,
                "ehetstar": het_res.ehetstar,
                "ohet": het_res.ohet,
                "ohet2": het_res.ohet2,
                "ehet": het_res.ehet,
            }
        ]
    )
    het_df.to_csv(f"{prefix}.het.csv", index=False)

    # Population-based analyses if pop-file supplied.
    if args.pop_file is not None:
        pops_full = _load_populations(args.pop_file, ra.sample_ids)
        pops_qc = pops_full[qc_result.keep_ind]

        # HW diagnostics.
        hw_res = popgen.hw_pops(
            genon=genon_np,
            depth=depth_np,
            populations=pops_qc,
            depth2K=depth2K_fn,
        )
        chrom = ra.chrom[qc_result.keep_snp]
        pos = ra.pos[qc_result.keep_snp]
        n_snp = chrom.shape[0]

        hw_rows = []
        for ipop, pop in enumerate(hw_res.popnames):
            for s in range(n_snp):
                hw_rows.append(
                    {
                        "population": pop,
                        "chrom": chrom[s],
                        "pos": pos[s],
                        "HWdis": hw_res.HWdis[ipop, s],
                        "l10LRT": hw_res.l10LRT[ipop, s],
                        "x2star": hw_res.x2star[ipop, s],
                        "l10pstar": hw_res.l10pstar[ipop, s],
                        "maf": hw_res.maf[ipop, s],
                        "l10pstar_pop": (
                            hw_res.l10pstar_pop[s]
                            if hw_res.l10pstar_pop is not None
                            else np.nan
                        ),
                    }
                )
        hw_df = pd.DataFrame(hw_rows)
        hw_df.to_csv(f"{prefix}.hw.csv", index=False)

        # Fst across populations.
        fst_res = popgen.fst_gbs(
            depth=depth_np,
            genon=genon_np,
            populations=pops_qc,
            depth2K=depth2K_fn,
            varadj=0,
        )
        fst_df = pd.DataFrame(
            {
                "chrom": chrom,
                "pos": pos,
                "Fst": fst_res.fst,
            }
        )
        fst_df.to_csv(f"{prefix}.fst.csv", index=False)

        # DAPC.
        dapc_res = popgen.dapc_from_genotypes(
            genon=genon_np,
            p=p_np,
            populations=pops_qc,
            sample_ids=kept_ids,
            n_pca=None,
            perc_pca=90.0,
        )
        n_pc = dapc_res.pc_scores.shape[1]
        n_disc = dapc_res.lda_scores.shape[1]
        dapc_data = {
            "sample_id": dapc_res.sample_ids,
            "pop": dapc_res.populations,
        }
        for i in range(n_pc):
            dapc_data[f"PC{i+1}"] = dapc_res.pc_scores[:, i]
        for j in range(n_disc):
            dapc_data[f"LD{j+1}"] = dapc_res.lda_scores[:, j]
        dapc_df = pd.DataFrame(dapc_data)
        dapc_df.to_csv(f"{prefix}.dapc.csv", index=False)

    # Pedigree-based parentage checks if ped-file supplied.
    if args.ped_file is not None:
        ped_df = pd.read_csv(args.ped_file)
        relF = args.rel_thresh_f if args.rel_thresh_f is not None else args.rel_thresh
        relM = args.rel_thresh_m if args.rel_thresh_m is not None else args.rel_thresh

        ped_out = ped.check_parents(
            ped_df=ped_df,
            sample_ids=ra.sample_ids,
            keep_ind_mask=qc_result.keep_ind,
            G5_diag=np.asarray(G5d),
            grm_op=op,
            depth_np=depth_np,
            genon_np=genon_np,
            p_np=p_np,
            depth2K=depth2K_fn,
            rel_threshF=relF,
            rel_threshM=relM,
            emm_thresh=args.emm_thresh,
            mindepth_mm=args.mindepth_mm,
        )
        ped_out.to_csv(f"{prefix}.ped.csv", index=False)


def cmd_inbreed(args: argparse.Namespace) -> None:
    """Compute depth-corrected inbreeding coefficients via G5 diagonal."""
    ra = io.read_ra_tab(args.ra_tab)
    qc_result = qc.run_qc(
        ra,
        sampdepth_thresh=args.sampdepth_thresh,
        snpdepth_thresh=args.snpdepth_thresh,
        maf_thresh=args.maf_thresh,
        pmethod=args.pmethod,
    )

    op = grm.build_grm_operator(
        depth=qc_result.depth,
        genon=qc_result.genon,
        p=qc_result.p,
        dmodel=args.depth_model,
        dparam=args.depth_param,
    )
    G5d = op.diag_G5()
    F_hat = G5d - 1.0

    kept_ids = np.array(ra.sample_ids)[qc_result.keep_ind]
    out_df = pd.DataFrame(
        {"sample_id": kept_ids, "F_hat": np.asarray(F_hat, dtype=np.float64)}
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "diag":
        cmd_diag(args)
    elif args.command == "block":
        cmd_block(args)
    elif args.command == "het":
        cmd_het(args)
    elif args.command == "hw":
        cmd_hw(args)
    elif args.command == "inbreed":
        cmd_inbreed(args)
    elif args.command == "fst":
        cmd_fst(args)
    elif args.command == "ped":
        cmd_ped(args)
    elif args.command == "dapc":
        cmd_dapc(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "merge-ra":
        cmd_merge_ra(args)
    elif args.command == "fit-depth":
        cmd_fit_depth(args)
    elif args.command == "snpselect":
        df = pd.read_csv(args.pos_csv)
        cols = {c.lower(): c for c in df.columns}
        if "chrom" in cols:
            chrom_col = cols["chrom"]
        elif "chromosome" in cols:
            chrom_col = cols["chromosome"]
        else:
            raise SystemExit("Position file must have 'chrom' or 'chromosome' column.")
        if "pos" in cols:
            pos_col = cols["pos"]
        elif "position" in cols:
            pos_col = cols["position"]
        else:
            raise SystemExit("Position file must have 'pos' or 'position' column.")

        chrom = df[chrom_col].to_numpy()
        pos = df[pos_col].to_numpy()
        chromuse = args.chrom_use if args.chrom_use else None
        pairs = popgen.snpselection(
            chromosome=chrom,
            position=pos,
            nsnpperchrom=args.nsnp_per_chrom,
            seltype=args.seltype,
            snpsubset=None,
            chromuse=chromuse,
            randseed=args.randseed,
        )
        out_df = pd.DataFrame({"snp1": pairs[:, 0], "snp2": pairs[:, 1]})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "ped-best":
        cmd_ped_best(args)
    elif args.command == "vcf2kgd":
        cmd_vcf2kgd(args)
    elif args.command == "snpselect":
        df = pd.read_csv(args.pos_csv)
        cols = {c.lower(): c for c in df.columns}
        if "chrom" in cols:
            chrom_col = cols["chrom"]
        elif "chromosome" in cols:
            chrom_col = cols["chromosome"]
        else:
            raise SystemExit("Position file must have 'chrom' or 'chromosome' column.")
        if "pos" in cols:
            pos_col = cols["pos"]
        elif "position" in cols:
            pos_col = cols["position"]
        else:
            raise SystemExit("Position file must have 'pos' or 'position' column.")

        chrom = df[chrom_col].to_numpy()
        pos = df[pos_col].to_numpy()
        chromuse = args.chrom_use if args.chrom_use else None
        pairs = popgen.snpselection(
            chromosome=chrom,
            position=pos,
            nsnpperchrom=args.nsnp_per_chrom,
            seltype=args.seltype,
            snpsubset=None,
            chromuse=chromuse,
            randseed=args.randseed,
        )
        out_df = pd.DataFrame({"snp1": pairs[:, 0], "snp2": pairs[:, 1]})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
    elif args.command == "plot-dapc":
        plots.plot_dapc_scatter(
            dapc_csv=args.dapc_csv,
            out_png=args.out,
            x_axis=args.x_axis,
            y_axis=args.y_axis,
        )
    elif args.command == "plot-fst":
        plots.plot_fst_manhattan(
            fst_csv=args.fst_csv,
            out_png=args.out,
        )
    elif args.command == "plot-hw":
        plots.plot_hw_dis_maf(
            hw_csv=args.hw_csv,
            out_png=args.out,
            pop=args.population,
        )
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
