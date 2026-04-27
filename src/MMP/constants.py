"""
Shared constants for MMP GRN analyses.
Imported by both network_analysis/script.py and notebook.ipynb.
"""

import pickle
import pandas as pd

GWAS_PATH   = '/vol/projects/CIIM/agentic_central/datalake/biomni/gwas_catalog.pkl'
TF_ALL_PATH = '/vol/projects/jnourisa/genernbi/resources/grn_benchmark/prior/tf_all.csv'
MSIGDB_C5   = '/vol/projects/CIIM/agentic_central/datalake/biomni/msigdb_human_c5_ontology_geneset.parquet'

# ── Immune cell-type marker TFs (dataset-independent) ────────────────────────
# 30 canonical PBMC lineage markers (manually curated from literature/CellTypist)
# + PAX5 (B cell identity) + BCL6 (germinal centre B cell, upstream of plasma cell)
IMMUNE_CORE_TFS = [
    # Myeloid / DC
    'SPI1', 'IRF8', 'CEBPA', 'CEBPB', 'MAFB',
    # B cell lineage (incl. plasma cell commitment)
    'PAX5', 'EBF1', 'BCL6', 'BACH2', 'TCF4', 'IRF4', 'PRDM1', 'XBP1',
    # T cell subsets
    'TBX21', 'GATA3', 'RORC', 'FOXP3', 'BATF',
    'TCF7', 'LEF1', 'BCL11B', 'ZEB2', 'ETS1', 'KLF2',
    # CD8 / NK
    'EOMES', 'RUNX3', 'ZNF683',
    # Pan-lymphocyte / progenitor
    'IKZF1', 'IKZF2', 'RUNX1', 'IRF7',
]  # n=32

# ── MMP disease TFs — GWAS + literature ──────────────────────────────────────
# Literature TFs: directly implicated in MMP/MM pathogenesis with citation.
# Note: EZH2 and KDM6A are important MM epigenetic regulators but are NOT in
# tf_all (they are histone modifiers, not canonical TFs) and are excluded.
MMP_LITERATURE_TFS = {
    # ── MYC pathway (translocations / amplifications) ──────────────────────
    'MYC': (
        'Translocation to Ig loci (~15–20% MM at progression); drives proliferation. '
        'Chesi et al. Blood 2008, DOI:10.1182/blood-2007-11-124644'
    ),
    'MYCN': (
        'Amplified in high-risk/relapsed MM; promotes proliferation. '
        'Beltran et al. Cancer Cell 2011, DOI:10.1016/j.ccr.2011.09.022'
    ),
    'MAX': (
        'Obligate MYC heterodimer partner; MYC oncogenic activity requires MAX. '
        'Blackwood & Eisenman Science 1991; Conacci-Sorrell et al. Cold Spring Harb Perspect Med 2014'
    ),
    # ── Ikaros family (drug targets / deletions) ───────────────────────────
    'IKZF1': (
        'Del(13q)/del(7p) in MM; target of thalidomide analogues via CRBN ubiquitin ligase. '
        'Mulligan et al. Cancer Res 2007, DOI:10.1158/0008-5472.CAN-07-0328'
    ),
    'IKZF3': (
        'Degraded by lenalidomide/pomalidomide via CRBN; mediates anti-MM activity. '
        'Kronke et al. Science 2014, DOI:10.1126/science.1244851'
    ),
    # ── IRF4 (plasma cell identity + direct MM drug mechanism) ────────────
    'IRF4': (
        'MM cell survival factor; degraded by lenalidomide/pomalidomide via CRBN. '
        'Kronke et al. Science 2014, DOI:10.1126/science.1244851; '
        'Pathak et al. Blood 2011, DOI:10.1182/blood-2011-01-329136'
    ),
    # ── NF-κB pathway (recurrent somatic mutations) ────────────────────────
    'NFKB1': (
        'NF-κB activated in ~20% MM via TRAF3/CYLD inactivating mutations. '
        'Demchenko et al. J Clin Oncol 2010, DOI:10.1200/JCO.2009.26.6353'
    ),
    'NFKB2': (
        'Non-canonical NF-κB; NFKB2 rearrangements recurrent in MM. '
        'Annunziata et al. Cancer Cell 2007, DOI:10.1016/j.ccr.2007.07.004'
    ),
    'RELA': (
        'Canonical NF-κB subunit; IL-6-induced MM cell survival via MAPK. '
        'Hideshima et al. Blood 2002, DOI:10.1182/blood.v99.6.2070'
    ),
    'RELB': (
        'Non-canonical NF-κB; activated by TRAF3 deletion in MM. '
        'Demchenko et al. J Clin Oncol 2010, DOI:10.1200/JCO.2009.26.6353'
    ),
    # ── MAF family (translocation oncoproteins) ────────────────────────────
    'MAF': (
        't(14;16)(q32;q23) translocation in ~5% MM; upregulates CCND2 and DKK1. '
        'Hurt et al. Blood 2004, DOI:10.1182/blood-2004-04-1342'
    ),
    'MAFB': (
        't(14;20)(q32;q12) translocation in ~2% MM; oncogenic MYC-like function. '
        'Keats et al. Blood 2005, DOI:10.1182/blood-2005-04-1825'
    ),
    # ── Microenvironment / signalling ──────────────────────────────────────
    'STAT3': (
        'Constitutively activated via IL-6 in MM; promotes survival and drug resistance. '
        'Catlett-Falcone et al. Immunity 1999, DOI:10.1016/S1074-7613(00)80031-4'
    ),
    'HIF1A': (
        'Activated in hypoxic bone marrow niche; promotes VEGF and bortezomib resistance. '
        'Zhang et al. Blood 2009, DOI:10.1182/blood-2009-01-200667'
    ),
    # ── Tumour suppressors ─────────────────────────────────────────────────
    'TP53': (
        'Del(17p)/TP53 mutation in ~10% newly diagnosed and >50% relapsed MM; high-risk marker. '
        'Teoh et al. Leukemia 2014, DOI:10.1038/leu.2013.265'
    ),
}


def get_mmp_gwas_tfs(pvalue_mlog_min: float = 7.3) -> set:
    """Return MMP-associated TF symbols from GWAS Catalog.

    Pipeline (mirrors IBD): genome-wide significant (p < 5e-8), genic SNPs only,
    excludes intergenic/synonymous variants, intersects with tf_all, then filters
    to TFs annotated to immune-related GO biological processes (MSigDB C5).
    Returns {KLF2, PRDM1, SP140, SP3}.
    """
    import ast

    gwas = pickle.load(open(GWAS_PATH, 'rb'))
    mask = gwas['DISEASE/TRAIT'].str.lower().str.contains(
        'multiple myeloma|plasma cell myeloma|myeloma|mgus|monoclonal gammopathy|smoldering',
        na=False,
    )
    sub = gwas[mask].copy()
    sub = sub[sub['PVALUE_MLOG'] >= pvalue_mlog_min]
    sub = sub[sub['INTERGENIC'] == 0]
    sub = sub[~sub['CONTEXT'].str.lower().str.contains(
        'intergenic_variant|synonymous_variant', na=True)]

    def parse_genes(s):
        if pd.isna(s):
            return []
        return [g.strip() for g in s.replace(' - ', ', ').split(',')
                if g.strip() and g.strip() != 'NR']

    genes = set()
    for col in ['REPORTED GENE(S)', 'MAPPED_GENE']:
        sub[col].dropna().apply(parse_genes).apply(genes.update)

    tf_all = set(
        pd.read_csv(TF_ALL_PATH).iloc[:, 0].astype(str).str.strip()
    )
    gwas_tfs = genes & tf_all

    # Immune GO filter (MSigDB C5) — same as IBD pipeline
    c5 = pd.read_parquet(MSIGDB_C5)
    immune_gsets = c5[c5['chromosome_id'].str.contains(
        'IMMUNE|LYMPHOCYTE|LEUKOCYTE|TCELL|BCELL|MYELOID|INNATE|ADAPTIVE|'
        'DEFENSE|INFLAMMATORY|NK_CELL|DENDRITIC|PLASMA_CELL',
        na=False, case=False,
    )]
    immune_genes = set()
    for val in immune_gsets['geneSymbols'].dropna():
        try:
            immune_genes.update(ast.literal_eval(val))
        except Exception:
            pass

    return gwas_tfs & immune_genes


def get_mmp_tfs() -> set:
    """Return full MMP disease TF set: GWAS (immune GO-filtered) + literature."""
    return get_mmp_gwas_tfs() | set(MMP_LITERATURE_TFS.keys())
