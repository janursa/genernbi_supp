"""
script.py — Process MMP (Multiple Myeloma Precursor) multiome data for GRN benchmark.

Steps:
  1. merge_peaks  : bedtools consensus merge of raw ATAC peaks (~1.3M → ~271k)
  2. preprocess   : QC, cell-type annotation, normalization, protocol split

Reads from:  task_grn_inference/resources/datasets_raw/
  - MMP_rna.h5ad           (101k cells × 36k genes, raw counts, Ensembl IDs)
  - MMP_atac.h5ad          (101k cells × ~1.3M peaks, outer-join raw counts)

Writes to:   task_grn_inference/resources/results/MMP/
  - MMP_atac_merged.h5ad       (consensus peaks intermediate)
  - MMP_rna_processed.h5ad
  - MMP_atac_processed.h5ad
  - MMP_{dogma,multiome}_{rna,atac}_processed.h5ad

Requires bedtools at /vol/biotools/bin/bedtools (bind /vol/biotools into singularity).
"""

import gzip
import os
import sys
import subprocess
import tempfile
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import argparse
from bisect import bisect_right

TASK_GRN_DIR = '/home/jnourisa/projs/ongoing/task_grn_inference'
GENERNBI_DIR = '/home/jnourisa/projs/ongoing/geneRNBI'
CIIM_TOOLS   = '/vol/projects/CIIM/agentic_central/tools/ciim/code'
BEDTOOLS     = '/vol/biotools/bin/bedtools'

sys.path.insert(0, TASK_GRN_DIR)
sys.path.insert(0, CIIM_TOOLS)
from src.process_data.helper_data import normalize_func
from genomics import identify_counts_layer, annotate_celltype_celltypist

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--out_dir', type=str, default=None)
parser.add_argument('--annotation_file', type=str,
                    default=os.path.join(TASK_GRN_DIR, 'resources/supp_data/gencode.v47.annotation.gtf.gz'))
args, _ = parser.parse_known_args()

DATASETS_RAW = os.path.join(TASK_GRN_DIR, 'resources/datasets_raw')
RESULTS_MMP  = os.path.join(TASK_GRN_DIR, 'resources/results/MMP')
OUT_DIR      = args.out_dir if args.out_dir else RESULTS_MMP
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Bedtools consensus peak merge
# ---------------------------------------------------------------------------

def _parse_peak_name(name):
    """Parse 'seqname:start-end' → (seqname, int(start), int(end))."""
    seqname, coords = name.split(':', 1)
    start, end = coords.split('-')
    return seqname, int(start), int(end)


def _run_bedtools_merge(var_names, tmp_dir):
    raw_bed    = os.path.join(tmp_dir, 'all_peaks.bed')
    sorted_bed = os.path.join(tmp_dir, 'all_peaks_sorted.bed')
    merged_bed = os.path.join(tmp_dir, 'consensus_peaks.bed')

    print(f'Writing {len(var_names):,} peaks to BED...', flush=True)
    with open(raw_bed, 'w') as f:
        for name in var_names:
            try:
                seqname, start, end = _parse_peak_name(name)
                f.write(f'{seqname}\t{start}\t{end}\t{name}\n')
            except Exception:
                pass

    subprocess.run(f'sort -k1,1 -k2,2n {raw_bed} > {sorted_bed}', shell=True, check=True)

    result = subprocess.run(
        [BEDTOOLS, 'merge', '-i', sorted_bed],
        capture_output=True, text=True, check=True
    )
    with open(merged_bed, 'w') as f:
        f.write(result.stdout)

    n_merged = result.stdout.count('\n')
    print(f'Bedtools merge: {len(var_names):,} → {n_merged:,} consensus peaks', flush=True)
    return merged_bed


def _build_mapping(var_names, merged_bed):
    merged_by_chrom = {}
    merged_peaks = []
    with open(merged_bed) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            seqname, start, end = parts[0], int(parts[1]), int(parts[2])
            merged_idx = len(merged_peaks)
            merged_peaks.append(f'{seqname}:{start}-{end}')
            merged_by_chrom.setdefault(seqname, []).append((start, end, merged_idx))

    for chrom in merged_by_chrom:
        merged_by_chrom[chrom].sort()

    print(f'Building peak mapping for {len(var_names):,} original peaks...', flush=True)
    mapping = np.full(len(var_names), -1, dtype=np.int32)
    for i, name in enumerate(var_names):
        try:
            seqname, start, end = _parse_peak_name(name)
        except Exception:
            continue
        intervals = merged_by_chrom.get(seqname, [])
        if not intervals:
            continue
        starts = [iv[0] for iv in intervals]
        pos = bisect_right(starts, start) - 1
        if pos >= 0:
            iv_start, iv_end, iv_idx = intervals[pos]
            if iv_start <= start and end <= iv_end:
                mapping[i] = iv_idx

    mapped = (mapping >= 0).sum()
    print(f'Mapped {mapped:,}/{len(var_names):,} peaks ({100*mapped/len(var_names):.1f}%)', flush=True)
    return merged_peaks, mapping


def _aggregate_counts(X, mapping, n_merged):
    n_cells, n_orig = X.shape
    X_csr = sp.csr_matrix(X)
    valid = mapping >= 0
    rows  = mapping[valid].astype(np.int32)
    cols  = np.where(valid)[0].astype(np.int32)
    data  = np.ones(rows.shape[0], dtype=np.float32)
    M     = sp.csr_matrix((data, (rows, cols)), shape=(n_merged, n_orig))
    print('Aggregating counts via sparse matrix multiply...', flush=True)
    return (X_csr @ M.T).tocsr()


def merge_peaks(input_atac, output_atac):
    """Bedtools consensus merge raw ATAC peaks. Skipped if output already exists."""
    if os.path.exists(output_atac):
        print(f'merge_peaks: {output_atac} already exists, skipping.', flush=True)
        return

    print(f'Loading {input_atac}...', flush=True)
    adata = ad.read_h5ad(input_atac)
    print(f'Loaded: {adata.shape[0]:,} cells × {adata.shape[1]:,} peaks', flush=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        merged_bed          = _run_bedtools_merge(adata.var_names.tolist(), tmp_dir)
        merged_peaks, mapping = _build_mapping(adata.var_names.tolist(), merged_bed)
        X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
        X_merged = _aggregate_counts(X, mapping, len(merged_peaks))

    peak_df = pd.DataFrame(index=merged_peaks)
    peak_df['seqname'] = [p.split(':')[0] for p in merged_peaks]
    coords = [p.split(':')[1].split('-') for p in merged_peaks]
    peak_df['start']  = [int(c[0]) for c in coords]
    peak_df['end']    = [int(c[1]) for c in coords]
    peak_df['ranges'] = [f'{c[0]}-{c[1]}' for c in coords]
    peak_df['strand'] = '+'

    adata_merged = ad.AnnData(
        X=X_merged.astype(np.float32),
        obs=adata.obs.copy(),
        var=peak_df,
    )
    adata_merged.var.index.name = 'peak_id'

    print(f'Saving to {output_atac}...', flush=True)
    adata_merged.write_h5ad(output_atac)
    print(f'Done. Shape: {adata_merged.shape[0]:,} × {adata_merged.shape[1]:,}', flush=True)


# ---------------------------------------------------------------------------
# Step 2: GTF interval mapping
# ---------------------------------------------------------------------------

def add_interval_from_gtf(adata, gtf_path):
    print(f'Parsing GTF: {gtf_path}', flush=True)
    gene_id_to_info = {}
    open_fn = gzip.open if gtf_path.endswith('.gz') else open
    with open_fn(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'gene':
                continue
            chrom, start, end = fields[0], fields[3], fields[4]
            attrs = fields[8]
            gene_id, gene_name = None, None
            for attr in attrs.split(';'):
                attr = attr.strip()
                if attr.startswith('gene_id'):
                    gene_id = attr.split(' ', 1)[1].strip().strip('"').split('.')[0]
                elif attr.startswith('gene_name'):
                    gene_name = attr.split(' ', 1)[1].strip().strip('"')
            if gene_id:
                gene_id_to_info[gene_id] = {
                    'gene_name': gene_name,
                    'interval':  f'{chrom}:{start}-{end}',
                }

    var_ids = adata.var_names.str.split('.').str[0]
    adata.var['gene_name'] = var_ids.map(lambda g: gene_id_to_info.get(g, {}).get('gene_name', g))
    adata.var['interval']  = var_ids.map(lambda g: gene_id_to_info.get(g, {}).get('interval'))

    n_gene = adata.var['gene_name'].notna().sum()
    n_int  = adata.var['interval'].notna().sum()
    print(f'GTF: mapped gene_name {n_gene}/{adata.n_vars}, interval {n_int}/{adata.n_vars}', flush=True)
    return adata


# ---------------------------------------------------------------------------
# obs / var formatting
# ---------------------------------------------------------------------------

def format_obs(adata):
    obs = adata.obs.copy()
    obs['barcode']  = obs['barcode_orig'].astype(str)
    obs['donor_id'] = obs['donor_id'].astype(str)
    obs['batch']    = obs['batch'].astype(str)
    obs['protocol'] = obs['protocol'].astype(str)

    def parse_timepoint(row):
        tp = str(row.get('timepoint', 'unknown'))
        if tp != 'unknown':
            return tp
        sid   = str(row.get('sample_id', ''))
        parts = sid.upper().split('_')
        if 'BL' in parts and 'ES' not in parts:
            return 'BL'
        if 'ES' in parts and 'BL' not in parts:
            return 'ES'
        return 'unknown'
    obs['timepoint'] = obs.apply(parse_timepoint, axis=1)

    disease_map = {'multiple_myeloma_precursor': 'mmp'}
    obs['disease']      = obs['disease'].map(lambda d: disease_map.get(str(d).lower(), str(d).lower()))
    obs['perturbation'] = 'none'
    obs['condition']    = 'mmp'
    obs['is_control']   = False
    obs['cell_type']    = 'unknown'

    keep = ['barcode', 'donor_id', 'batch', 'protocol', 'timepoint',
            'disease', 'perturbation', 'condition', 'is_control', 'cell_type']
    adata.obs = obs[keep].copy()
    adata.obs.index = adata.obs_names
    adata.obs.index.name = 'obs_id'
    return adata


def format_atac_var(adata):
    STANDARD_CHR = {f'chr{i}' for i in list(range(1, 23)) + ['X', 'Y', 'M']}
    peak_names = adata.var_names.tolist()
    seqnames, starts, ends = [], [], []
    for p in peak_names:
        try:
            seq, coords = p.split(':', 1)
            s, e = coords.split('-')
            seqnames.append(seq); starts.append(int(s)); ends.append(int(e))
        except Exception:
            seqnames.append(''); starts.append(0); ends.append(0)

    adata.var['seqname'] = seqnames
    adata.var['start']   = starts
    adata.var['end']     = ends
    adata.var['ranges']  = [f'{s}-{e}' for s, e in zip(starts, ends)]
    adata.var['strand']  = '+'

    before = adata.n_vars
    adata  = adata[:, adata.var['seqname'].isin(STANDARD_CHR)].copy()
    print(f'ATAC: kept {adata.n_vars:,}/{before:,} peaks on standard chromosomes', flush=True)

    adata.var.index = (adata.var['seqname'].astype(str) + ':' +
                       adata.var['start'].astype(str) + '-' +
                       adata.var['end'].astype(str))
    adata.var.index.name = 'peak_id'
    return adata


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------

def qc_rna(adata, min_genes=200, min_cells=10):
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f'RNA after QC: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes', flush=True)
    return adata


def qc_atac(adata, min_fragments=500, max_fragments=100_000,
             min_features=200, max_features=100_000):
    ncount   = np.array(adata.X.sum(axis=1)).flatten()
    nfeature = np.array((adata.X > 0).sum(axis=1)).flatten()
    adata.obs['nCount_ATAC']   = ncount
    adata.obs['nFeature_ATAC'] = nfeature
    mask = (
        (ncount   >= min_fragments) & (ncount   <= max_fragments) &
        (nfeature >= min_features)  & (nfeature <= max_features)
    )
    print(f'ATAC: keeping {mask.sum():,}/{adata.n_obs:,} cells after QC', flush=True)
    return adata[mask].copy()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def add_metadata(adata):
    adata.uns['dataset_id']          = 'MMP'
    adata.uns['dataset_name']        = 'Multiple Myeloma Precursor (MMP)'
    adata.uns['dataset_summary']     = (
        'Multiome (scRNA+scATAC) of bone marrow from multiple myeloma precursor '
        'patients (MGUS/SMM/MM). 12 samples, 3 batches (SD-2520, SD-2589, SD-2752), '
        '10x Multiome and DOGMA-seq protocols. GSE311602.'
    )
    adata.uns['dataset_description'] = adata.uns['dataset_summary']
    adata.uns['data_reference']      = 'GSE311602'
    adata.uns['data_url']            = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE311602'
    adata.uns['dataset_organism']    = 'human'
    adata.uns['normalization_id']    = 'lognorm'
    return adata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Step 1: merge peaks (skipped if already done)
merge_peaks(
    input_atac  = os.path.join(DATASETS_RAW, 'MMP_atac.h5ad'),
    output_atac = os.path.join(RESULTS_MMP, 'MMP_atac_merged.h5ad'),
)

# Step 2: load merged ATAC + RNA
print('Loading MMP_rna.h5ad...', flush=True)
adata_rna = ad.read_h5ad(f'{DATASETS_RAW}/MMP_rna.h5ad')
print(f'RNA loaded: {adata_rna.shape}', flush=True)

print('Loading MMP_atac_merged.h5ad...', flush=True)
adata_atac = ad.read_h5ad(f'{RESULTS_MMP}/MMP_atac_merged.h5ad')
print(f'ATAC loaded: {adata_atac.shape}', flush=True)

adata_rna  = format_obs(adata_rna)
adata_atac = format_obs(adata_atac)
adata_rna  = qc_rna(adata_rna)
adata_atac = qc_atac(adata_atac)
adata_atac = format_atac_var(adata_atac)

common = set(adata_rna.obs_names) & set(adata_atac.obs_names)
print(f'Common cell IDs after QC: {len(common):,}', flush=True)
adata_rna  = adata_rna [adata_rna .obs_names.isin(common)].copy()
adata_atac = adata_atac[adata_atac.obs_names.isin(common)].copy()
print(f'RNA  after barcode match: {adata_rna.shape}', flush=True)
print(f'ATAC after barcode match: {adata_atac.shape}', flush=True)

adata_rna.X = sp.csr_matrix(adata_rna.X.astype(np.float32))
adata_rna   = add_interval_from_gtf(adata_rna, args.annotation_file)

no_symbol = adata_rna.var['gene_name'].isna()
n_before  = adata_rna.n_vars
adata_rna = adata_rna[:, ~no_symbol].copy()
print(f'RNA: dropped {no_symbol.sum()} genes with no gene symbol in GTF ({n_before} → {adata_rna.n_vars})', flush=True)
adata_rna.var_names = adata_rna.var['gene_name'].astype(str)
adata_rna.var_names_make_unique()
adata_rna.var.drop(columns=['gene_name'], inplace=True)

ensg_mask = adata_rna.var_names.str.startswith('ENSG')
n_before  = adata_rna.n_vars
adata_rna = adata_rna[:, ~ensg_mask].copy()
print(f'RNA after gene symbol rename: {adata_rna.shape} (dropped {ensg_mask.sum()} unmapped Ensembl genes)', flush=True)

print('Running CellTypist annotation...', flush=True)
identify_counts_layer(adata_rna)
annotate_celltype_celltypist(adata_rna)
adata_rna.obs['cell_type'] = 'pbmc'
print(f'Cell types: {adata_rna.obs["cell_type"].value_counts().to_dict()}', flush=True)

adata_atac.obs['cell_type'] = 'pbmc'
for col in ['CT_Major', 'CT_Major_percell', 'CT_Minor', 'CT_Minor_percell', 'leiden']:
    if col in adata_rna.obs.columns:
        adata_atac.obs[col] = adata_atac.obs_names.map(adata_rna.obs[col].to_dict()).fillna('unknown')

adata_rna = normalize_func(adata_rna)
adata_atac.X = sp.csr_matrix(adata_atac.X.astype(np.float32))

adata_rna  = add_metadata(adata_rna)
adata_atac = add_metadata(adata_atac)

for col in ['donor_id', 'disease', 'perturbation', 'condition', 'cell_type', 'batch']:
    if col in adata_rna.obs.columns:
        adata_rna.obs[col]  = adata_rna.obs[col].astype(str)
    if col in adata_atac.obs.columns:
        adata_atac.obs[col] = adata_atac.obs[col].astype(str)

out_rna  = os.path.join(OUT_DIR, 'MMP_rna_processed.h5ad')
out_atac = os.path.join(OUT_DIR, 'MMP_atac_processed.h5ad')
print(f'Saving RNA  → {out_rna}',  flush=True)
adata_rna.write_h5ad(out_rna)
print(f'Saving ATAC → {out_atac}', flush=True)
adata_atac.write_h5ad(out_atac)

# Protocol-split versions
DOGMA_BATCHES    = ['SD-2589', 'SD-2752']
MULTIOME_BATCHES = ['SD-2520']

for batches, tag in [(DOGMA_BATCHES, 'dogma'), (MULTIOME_BATCHES, 'multiome')]:
    mask_rna  = adata_rna .obs['batch'].isin(batches)
    mask_atac = adata_atac.obs['batch'].isin(batches)
    sub_rna   = adata_rna [mask_rna ].copy()
    sub_atac  = adata_atac[mask_atac].copy()

    if tag == 'dogma':
        top3_donors = sub_rna.obs['donor_id'].value_counts().head(3).index.tolist()
        print(f'DOGMA: keeping top-3 donors {top3_donors}', flush=True)
        donor_mask = sub_rna.obs['donor_id'].isin(top3_donors)
        sub_rna  = sub_rna [donor_mask].copy()
        sub_atac = sub_atac[sub_rna.obs_names].copy()
        if sub_rna.n_obs > 25000:
            rng = np.random.default_rng(42)
            idx = rng.choice(sub_rna.n_obs, size=25000, replace=False)
            idx.sort()
            sub_rna  = sub_rna [idx].copy()
            sub_atac = sub_atac[sub_rna.obs_names].copy()
        print(f'DOGMA after subsample: {sub_rna.n_obs:,} cells', flush=True)

    sub_rna .uns['dataset_id'] = f'MMP_{tag}'
    sub_atac.uns['dataset_id'] = f'MMP_{tag}'
    out_r = os.path.join(OUT_DIR, f'MMP_{tag}_rna_processed.h5ad')
    out_a = os.path.join(OUT_DIR, f'MMP_{tag}_atac_processed.h5ad')
    print(f'Saving {tag} RNA  ({sub_rna.n_obs:,} cells) → {out_r}',  flush=True)
    sub_rna.write_h5ad(out_r)
    print(f'Saving {tag} ATAC ({sub_atac.n_obs:,} cells) → {out_a}', flush=True)
    sub_atac.write_h5ad(out_a)

print(f'Done. RNA: {adata_rna.shape}  ATAC: {adata_atac.shape}', flush=True)
