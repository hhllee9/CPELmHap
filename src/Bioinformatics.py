import tabix
import bisect
import numpy as np



# Retrieve all CpG site positions within a specified genomic interval
def get_cpg_in_range(chr, start, end, cpg_sites_file):
    # Open the Tabix index for the reference CpG site file
    tbx = tabix.open(cpg_sites_file)
    
    # Query CpG sites in the specified interval
    results = tbx.query(str(chr), start, end)  
    cpg_positions = []
    for result in results:
        cpgsite= int(result[1])
        # Confirm the CpG site lies within the target interval
        if cpgsite >= start and cpgsite <= end:
            cpg_positions.append(cpgsite)
    
    return cpg_positions


def extract_reads_in_range(chr, start, end, df):
    # Identify reads overlapping the target region
    mask = (df[0] == chr) & (df[1] < end) & (df[2] > start)
    out = df.loc[mask].copy()
    if out.empty:
        return out, None, None


    # Compute the span of the overlapping reads using raw coordinates so CpGs outside the ROI are included
    
    cover_start = int(out[1].min())
    cover_end   = int(out[2].max())


    return out, cover_start, cover_end



# Construct a read vector aligned to the full CpG coordinate set
def construct_x(read_start, read_end, ms, cpg_all):
    t = len(cpg_all)
    x = [-1]*t
    if t == 0:
        return x

    # Locate the read coverage interval [L_all, R_all] within the full CpG coordinate set
    L_all = bisect.bisect_left(cpg_all, read_start)
    R_all = bisect.bisect_right(cpg_all, read_end) - 1
    if L_all > R_all or L_all >= t or R_all < 0:
        return x

    keep_len = R_all - L_all + 1

    # Normalize the methylation string and align it to keep_len (truncate if too long, pad with '.' to the right if too short)
    ms = [c for c in str(ms or "")]
    ms = [c if c in ('0','1','.') else '.' for c in ms]
    if len(ms) >= keep_len:
        seg = ms[:keep_len]
    else:
        seg = ms + ['.']*(keep_len - len(ms))

    # Write the aligned segment into the full CpG vector at indices [L_all, R_all]
    for i, m in enumerate(seg):
        j = L_all + i
        x[j] = 1 if m == '1' else (0 if m == '0' else -1)
    return x

def build_xobs_superset(df, cpg_all):
    """Generate xobs for all reads in df aligned to the full CpG coordinate set."""
    if df.empty:
        return []

    xobs = []
    for _, row in df.iterrows():
        rs = int(row[1]); re = int(row[2])
        ms = str(row[3]) if row[3] is not None else ""
        reps = int(row[4]) if row[4] is not None else 1

        x = construct_x(rs, re, ms, cpg_all)
        for _ in range(max(1, reps)):
            xobs.append(x.copy())
    return xobs

def build_xobs(xobs_superset, cpg_all, roi_start, roi_end ):



    """Slice the full-aligned xobs to the target region by column indices."""
    if not xobs_superset or not cpg_all:
        return []

    L = bisect.bisect_left(cpg_all, roi_start)
    R = bisect.bisect_right(cpg_all, roi_end) - 1
    if L > R or L >= len(cpg_all) or R < 0:
        return []

    return [row[L:R+1] for row in xobs_superset]



