#!/usr/bin/env python3
"""Validate local datasets, then launch existing evaluators with one frozen mode."""
import argparse
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from lib.minicpm.progressive_arbitration import CONFIG, MODE, BASELINE_MODE, DEFINITION, SYSTEM_DEFINITION

BENCHMARKS = ('ovo','streamingbench','streambench_v03','videomme','longvideobench','egoschema')


def path(env, default=None):
    value = os.environ.get(env, default)
    if not value:
        raise ValueError(f'Set {env} to the verified local dataset path; no guessed path is used.')
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f'{env}: missing {p}')
    return str(p)


def specification(bench, limit=0):
    prefix = 'main_experiments.minicpm_v46.'
    if bench == 'ovo':
        annotation = path('OVO_ANNO_PATH','data/ovo_bench/ovo_bench_new.json')
        videos = path('OVO_CHUNKED_DIR','data/ovo_bench/chunked_videos')
        rows = json.loads(Path(annotation).read_text())
        tasks = []
        for row in rows:
            names = ([f"{row['id']}_{j}.mp4" for j in range(len(row['test_info']))]
                     if 'test_info' in row else [f"{row['id']}.mp4"])
            tasks.extend(dict(video_path=str(Path(videos)/name)) for name in names)
        args = ['--anno_path',annotation,'--chunked_dir',videos,'--recent_frames_only','6',
                '--chunk_duration','1','--fps','1','--max_qa_tokens','256']
        if limit:
            args += ['--max_samples_total',str(limit)]
        return prefix+'ovo.eval_prism_exact_recent',args,tasks
    if bench == 'streamingbench':
        annotation = path('STREAMINGBENCH_ANNO_PATH','data/streamingbench/questions_real.json')
        videos = path('STREAMINGBENCH_VIDEO_DIR','data/streamingbench/videos')
        mod = importlib.import_module(prefix+'streamingbench.eval_baseline_dist')
        tasks,_ = mod._load_tasks(annotation,videos)
        return prefix+'streamingbench.eval_adaptive_dist', ['--anno-path',annotation,'--video-dir',videos,
            '--recent-frames-only','6','--context-time','-1','--max-samples',str(limit),'--max-qa-tokens','256'],tasks
    if bench == 'streambench_v03':
        annotation = path('STREAMBENCH_V03_ANNOTATIONS','data/streambench_v0_3/streaming_bench_v0.3.json')
        videos = path('STREAMBENCH_V03_DATA_ROOT','data/streambench_v0_3')
        mod = importlib.import_module(prefix+'streambench_v03.eval_streambench_v03_smoke')
        tasks = mod._load_tasks(Path(annotation),Path(videos),0,0,'',0)
        print('WARNING: open-ended samples use logged Exact Recent-6 fallback; this is NOT option arbitration on those samples.')
        return prefix+'streambench_v03.eval_streambench_v03_smoke',['--anno-path',annotation,'--data-root',videos,
            '--methods','prism','--max-videos','0','--max-questions',str(limit),'--recent-window','6','--max-new-tokens','256'],tasks
    if bench == 'videomme':
        annotation = path('VIDEOMME_ANNOTATION_PARQUET','data/video_mme/videomme/test-00000-of-00001.parquet')
        videos = path('VIDEOMME_VIDEO_DIR','data/video_mme/videos')
        mod = importlib.import_module(prefix+'videomme.eval_videomme_dist')
        tasks = mod._load_tasks(SimpleNamespace(annotation_parquet=annotation,video_dir=videos,max_samples=0))
        args = ['--annotation-parquet',annotation,'--video-dir',videos]
    elif bench == 'longvideobench':
        annotation = path('LVB_ANNOTATION_JSON','data/longvideobench/lvb_val.json')
        videos = path('LVB_DATA_ROOT','data/longvideobench')
        mod = importlib.import_module(prefix+'longvideobench.eval_longvideobench_dist')
        tasks = mod._load_tasks(SimpleNamespace(annotation_json=annotation,data_root=videos,require_gt=True,max_subtitle_chars=0,max_samples=0))
        args = ['--annotation-json',annotation,'--data-root',videos,'--max-subtitle-chars','0']
    else:
        annotation = path('EGOSCHEMA_ANNOTATION_JSON','reports/egoschema_subset_annotations.json')
        videos = path('EGOSCHEMA_VIDEO_DIR','data/egoschema/videos')
        mod = importlib.import_module(prefix+'egoschema.eval_egoschema_dist')
        tasks = mod._load_tasks(SimpleNamespace(annotation_json=annotation,video_dir=videos,max_samples=0))
        if len(tasks) != 500 or len({t['video_id'] for t in tasks}) != 500:
            raise ValueError('EgoSchema requires exactly 500 unique labeled subset videos')
        args = ['--annotation-json',annotation,'--video-dir',videos,'--dataset-name','Subset']
    return mod.__name__,args+['--recent-frames-only','6','--max-samples',str(limit),'--max-qa-tokens','256'],tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark',choices=BENCHMARKS)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--baseline',action='store_true')
    parser.add_argument('--max-samples',type=int,default=int(os.environ.get('MAX_SAMPLES','0')))
    args = parser.parse_args()
    module, extra, tasks = specification(args.benchmark,args.max_samples)
    if not tasks:
        raise ValueError('Empty dataset')
    missing = [t['video_path'] for t in tasks if not Path(t['video_path']).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing videos; first: {missing[:10]}')
    print(f'Validated {len(tasks)} tasks for {args.benchmark}',flush=True)
    if args.check:
        return
    mode = BASELINE_MODE if args.baseline or os.environ.get('PRISM_BASELINE') == '1' else MODE
    job = os.environ.get('SLURM_JOB_ID',str(time.time_ns()))
    output = ROOT/'reports/progressive_arbitration'/args.benchmark/('recent6_' if mode==BASELINE_MODE else 'prism_')
    output = output.with_name(output.name+job)
    output.mkdir(parents=True,exist_ok=False)
    env = dict(os.environ)
    env.update(MINICPM_ADAPTIVE_MODE=mode, STREAMBENCH_PRISM_MODE=mode,
        MINICPM_ADAPTIVE_MIN_WINDOW='6',MINICPM_ADAPTIVE_MID_WINDOW='6',MINICPM_ADAPTIVE_MAX_WINDOW='6',
        MINICPM_EXACT_RECENT_CANDIDATE_FPS='4.0',MINICPM_PSM_EXACT_RECENT_PRESERVE_SOURCE_IDS='1',
        QWEN_EXACT_RECENT_DECODE='0',MINICPM_DOWNSAMPLE_MODE='16x',MINICPM_MAX_SLICE_NUMS='1',
        MINICPM_SEED='42',PYTHONHASHSEED='42',ATTN_IMPLEMENTATION='sdpa')
    if args.benchmark in ('ovo','streamingbench'):
        extra += ['--adaptive-mode',mode,'--adaptive-min-window','6','--adaptive-mid-window','6','--adaptive-max-window','6']
    elif args.benchmark != 'streambench_v03':
        extra += ['--mode',mode]
    extra += ['--result_dir' if args.benchmark=='ovo' else '--output-dir',str(output)]
    processes = 1 if args.benchmark=='streambench_v03' else int(env.get('NUM_PROCESSES','4'))
    if processes > 1:
        command = [sys.executable,'-m','accelerate.commands.launch','--num_processes',str(processes),
            '--num_machines','1','--main_process_port',str(29000+int(job)%2000),'--multi_gpu','--mixed_precision','bf16','--module',module]
    else:
        command = [sys.executable,'-m',module]
    command += extra
    manifest = dict(config=asdict(CONFIG),mode=mode,benchmark=args.benchmark,command=command,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        latency_definition=DEFINITION,system_definition=SYSTEM_DEFINITION,
        environment={k:v for k,v in env.items() if k.startswith(('MINICPM_','ROCM_','MIOPEN_','SLURM_','HIP_','CUDA_','ATTN_','HF_HOME'))})
    (output/'run_config.json').write_text(json.dumps(manifest,indent=2))
    print('RESULT_DIR='+str(output),flush=True)
    start = time.perf_counter()
    with (output/'run.log').open('w') as log:
        process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            print(line,end='',flush=True)
            log.write(line)
            log.flush()
        status = process.wait()
    (output/'job_timing.json').write_text(json.dumps(dict(evaluator_wall_seconds=time.perf_counter()-start,exit_code=status)))
    if status:
        raise SystemExit(status)
    from main_experiments.tools.summarize_progressive_arbitration import summarize_directory
    summary = summarize_directory(output,args.benchmark)
    if summary['errors']:
        raise SystemExit(f"Evaluation recorded {summary['errors']} sample errors; inspect {output}/run.log")


if __name__=='__main__':
    main()
