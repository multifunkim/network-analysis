#!/usr/bin/env python3
"""
pipeline_steps1_6.py – SPARK Steps 1–6
• l-max default is now None → Step 2 uses MATLAB rule L ≤ K/2
"""

import os, re, glob, shlex, logging, argparse, subprocess

# ── helpers ─────────────────────────────────────────────────────────────
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='[pipeline] %(asctime)s %(message)s',
        handlers=[logging.StreamHandler()])

def parse_steps(arg):
    if arg.lower() in ('all',''): return set(range(1,7))
    s=set()
    for part in arg.split(','):
        m=re.match(r'(\d+)(?:-(\d+))?$',part.strip())
        if not m: raise ValueError(f"Invalid --steps: {part}")
        a,b=int(m.group(1)),int(m.group(2) or m.group(1))
        s.update(range(a,b+1))
    return s

def run_step(cmd, marker, do_run, step):
    if not do_run:
        logging.info(f"SKIP step {step}")
        return
    if os.path.exists(marker):
        logging.info(f"✔ step {step} done ({os.path.basename(marker)})")
    else:
        logging.info(f"→ step {step}: {' '.join(shlex.quote(c) for c in cmd)}")
        subprocess.check_call(cmd)

# ── main ────────────────────────────────────────────────────────────────
def main():
    setup_logger()
    p=argparse.ArgumentParser(description="SPARK pipeline Steps 1–6")
    p.add_argument('--fmri_path',        required=True)
    p.add_argument('--mask_path',        required=True)
    p.add_argument('--output_dir',       required=True)
    p.add_argument('--network_scales',   nargs=3, type=int, required=True,
                   metavar=('KMIN','KSTEP','KMAX'))
    p.add_argument('--l-max',            type=int, default=None,
                   help='cap L grid in Step 2 (default None → K/2)')
    p.add_argument('--subsample_factor', type=int, default=1)
    p.add_argument('--nb_samps',         type=int, default=30)
    p.add_argument('--block_window_length', nargs=3, type=int, default=[30,1,30])
    p.add_argument('--max_parallel_jobs', type=int, default=8)
    p.add_argument('--n_iter',           type=int, default=8)
    p.add_argument('--pvalue',           type=float, default=0.05)
    p.add_argument('--min_voxels',       type=int, default=30)
    p.add_argument('--steps',            type=str,  default='all')
    # pass-through args for Step 2 if you’d like:
    p.add_argument('--step2_extra', nargs=argparse.REMAINDER,
                   help='additional flags forwarded to step2_estimate_scale.py')
    args=p.parse_args()

    to_run=parse_steps(args.steps)
    wd=args.output_dir; os.makedirs(wd, exist_ok=True)

    # ---------- output filenames ---------------------------------------
    mat1=os.path.join(wd,'tseries.mat')
    mat2=os.path.join(wd,'scale.mat')
    boot_dir=os.path.join(wd,'boot')
    dict_dir=os.path.join(wd,'dict')
    mat5=os.path.join(wd,'clusters.mat')

    # ---------- Step 1 --------------------------------------------------
    cmd1=['python','step1_load_data.py',
          '--fmri',args.fmri_path,
          '--mask',args.mask_path,
          '--subsample',str(args.subsample_factor),
          '--out',mat1]
    run_step(cmd1, mat1, 1 in to_run, 1)

    # ---------- Step 2 --------------------------------------------------
    Kmin,Kstep,Kmax=args.network_scales
    cmd2=['python','step2_estimate_scale.py',
          '--tseries',mat1,
          '--k-min',str(Kmin),'--k-step',str(Kstep),'--k-max',str(Kmax),
          '--out',mat2]
    if args.l_max and args.l_max>0:
        cmd2 += ['--l-max', str(args.l_max)]
    if args.step2_extra:
        cmd2 += args.step2_extra                      # pass-through
    run_step(cmd2, mat2, 2 in to_run, 2)

    # ---------- Step 3 --------------------------------------------------
    last_boot=os.path.join(boot_dir,f'boot_{args.nb_samps-1:03d}.mat')
    cmd3=['python','step3_bootstrap.py',
          '--tseries',mat1,
          '--block-len',str(args.block_window_length[0]),
          '--n-boot',str(args.nb_samps),
          '--outdir',boot_dir]
    run_step(cmd3, last_boot, 3 in to_run, 3)

    # ---------- Step 4 --------------------------------------------------
    # Check that *all* expected dictionaries exist before skipping
    expected_dicts = [
        os.path.join(dict_dir, f'boot_{b:03d}_dict.mat')
        for b in range(args.nb_samps)
    ]
    done_count = sum(os.path.exists(f) for f in expected_dicts)
    all_done = done_count == len(expected_dicts)
    
    cmd4 = [
        'python', 'step4_dictionary.py',
        '--bootstrap_dir', boot_dir,
        '--scale', mat2,
        '--outdir', dict_dir,
        '--n-iter', str(args.n_iter),
        '--n-jobs', str(args.max_parallel_jobs)
    ]
    
    if all_done:
        logging.info(f"✔ step 4 done (all {len(expected_dicts)} dictionaries found)")
    else:
        logging.info(f"→ step 4: running dictionary learning "
                     f"({done_count}/{len(expected_dicts)} existing)")
        subprocess.check_call(cmd4)

    # ---------- Step 5 --------------------------------------------------
    dicts=sorted(glob.glob(os.path.join(dict_dir,'*_dict.mat')))
    cmd5=['python','step5_clustering.py','--dicts']+dicts+[
          '--scale',mat2,'--out',mat5]
    run_step(cmd5, mat5, 5 in to_run, 5)

    # ---------- Step 6 --------------------------------------------------
    subj=os.path.basename(os.path.normpath(wd))
    kmap_dir=os.path.join(wd,f'KMAP_{subj}'); os.makedirs(kmap_dir, exist_ok=True)
    kmap_path=os.path.join(kmap_dir,f'k_hubness_{subj}.nii.gz')
    cmd6=['python','step6_kmap_atoms.py',
          '--clusters',mat5,
          '--tseries',mat1,
          '--mask',args.mask_path,
          '--pvalue',str(args.pvalue),
          '--min_voxels',str(args.min_voxels),
          '--outdir',kmap_dir,
          '--subject_label',subj]
    run_step(cmd6, kmap_path, 6 in to_run, 6)

    logging.info("✅ Pipeline Steps 1–6 complete.  Results in %s", wd)

if __name__ == '__main__':
    main()
