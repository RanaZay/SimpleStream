"""Per-question timing aggregation, paired comparison, and cross-benchmark reports."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from lib.minicpm.progressive_arbitration import DEFINITION, SYSTEM_DEFINITION


def distribution(values):
    values = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not values:
        return dict(n=0,mean=None,median=None,p50=None,p90=None,p95=None,p99=None,std=None)
    def pct(q):
        pos=(len(values)-1)*q
        low=int(pos)
        return values[low]+(values[min(low+1,len(values)-1)]-values[low])*(pos-low)
    return dict(n=len(values),mean=statistics.mean(values),median=pct(.5),p50=pct(.5),
                p90=pct(.9),p95=pct(.95),p99=pct(.99),std=statistics.pstdev(values))


def records(directory):
    rows=[]
    files=sorted(directory.glob('rank_*/results_incremental.jsonl'))
    if files:
        for file in files:
            for line in file.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    else:
        file=directory/'streambench_v0_3_smoke_results.json'
        if not file.exists():
            raise FileNotFoundError(f'No per-sample results in {directory}')
        payload=json.loads(file.read_text())
        rows=payload if isinstance(payload,list) else payload.get('results',payload.get('rows',[]))
    out=[]
    for row in rows:
        children=row.get('test_info')
        if isinstance(children,list):
            for j,child in enumerate(children):
                out.append({**row,**child,'pair_key':f"{row.get('task')}:{row.get('id')}:{j}"})
        elif 'methods' in row:
            for child in row['methods']:
                out.append({**row,**child,'pair_key':str((row.get('video_path'),row.get('time'),row.get('question')))})
        else:
            out.append({**row,'pair_key':str(row.get('_key',row.get('question_id',row.get('id',row.get('_index')))))})
    keys=[r['pair_key'] for r in out]
    if len(keys)!=len(set(keys)):
        raise ValueError('Duplicate question keys; refusing mixed/stale output aggregation')
    return out


def profile(row):
    p=row.get('profile') or row.get('profile_metadata') or {}
    a=p.get('adaptive') or row.get('adaptive_metadata') or row.get('adaptive') or {}
    return p,a,a.get('timing') or p.get('progressive_arbitration') or {}


def correctness(row,benchmark):
    if benchmark=='streambench_v03':
        return None  # Official judging is separate; lexical scores are not accuracy.
    if row.get('error'):
        return False
    if isinstance(row.get('correct'),bool):
        return row['correct']
    if benchmark=='ovo':
        from lib.shared.recent_window import score_ovo_br,score_ovo_rec,score_yes_no
        task=row['task']
        response=row.get('response')
        if task=='REC':
            return bool(score_ovo_rec(response,row.get('count',row.get('gt'))))
        if task in ('SSR','CRR'):
            return bool(score_yes_no(response,row.get('type',row.get('gt'))))
        return bool(score_ovo_br(response,row['ground_truth']))
    return None


def accuracy(rows,benchmark):
    values=[correctness(r,benchmark) for r in rows]
    if any(v is None for v in values) or not values:
        return None
    if benchmark!='ovo':
        return 100*sum(values)/len(values)
    groups=[('EPM','HLD','ASI'),('OCR','OJR','ACR','STU','ATR','FPD'),('REC','SSR','CRR')]
    scores=[]
    for group in groups:
        tasks=[]
        for task in group:
            vals=[v for r,v in zip(rows,values) if r.get('task')==task]
            if vals:
                tasks.append(statistics.mean(vals))
        if tasks:
            scores.append(statistics.mean(tasks))
    return 100*statistics.mean(scores) if scores else None


def summarize_directory(directory,benchmark,baseline=None):
    directory=Path(directory)
    rows=records(directory)
    profiles=[profile(r) for r in rows]
    timed=[t for _,_,t in profiles if t]
    names=sorted({k for t in timed for k,v in t.items() if (k.endswith('_ms') or k.endswith('_count')) and not isinstance(v,bool)})
    stats={k:distribution([t.get(k) for t in timed]) for k in names}
    result=dict(benchmark_name=benchmark,num_samples=len(rows),num_timed_samples=len(timed),
        errors=sum(bool(r.get('error')) for r in rows),prism_accuracy=accuracy(rows,benchmark),
        baseline_accuracy=None,accuracy_delta=None,latency_definition=DEFINITION,system_definition=SYSTEM_DEFINITION,
        metrics=stats,baseline_pairing='not supplied',depth={},evaluated_depth={})
    for field,target in [('selected_memory_depth','depth'),('max_depth_evaluated','evaluated_depth')]:
        for j in range(4):
            selected=[t for t in timed if t.get(field)==j]
            result[target][str(j)]={k:distribution([t.get(k) for t in selected]) for k in
                ('PRISM_selection_latency_ms','PRISM_end_to_end_latency_ms','final_generation_ms','final_ttft_ms')}
            result[target][str(j)]['n']=len(selected)
    def mean(field):
        return distribution([t.get(field) for t in timed])['mean']
    result['mean_selected_memory_depth']=mean('selected_memory_depth')
    result['mean_historical_frames']=result['mean_selected_memory_depth']
    for j in range(4):
        result[f'depth{j}_fraction']=sum(t.get('selected_memory_depth')==j for t in timed)/max(len(timed),1)
    result['memory_trigger_rate']=sum(bool(a.get('memory_triggered')) for _,a,_ in profiles)/max(len(rows),1)
    result['unsupported_arbitration_samples']=sum(a.get('arbitration_supported') is False for _,a,_ in profiles)
    result['mean_final_frames']=mean('final_frame_count')
    result['mean_vision_tokens']=distribution([r.get('num_vision_tokens') for r in rows])['mean']
    result['gpu_peak_allocated_mb']=distribution([p.get('gpu_peak_allocated_mb') for p,_,_ in profiles])
    result['gpu_peak_reserved_mb']=distribution([p.get('gpu_peak_reserved_mb') for p,_,_ in profiles])
    distances=[d for _,a,_ in profiles for d in a.get('memory_distances',[])]
    result['accepted_memory_distance']=distribution(distances)
    hits=sum(t.get('clip_cache_hits',0) for t in timed)
    misses=sum(t.get('clip_cache_misses',0) for t in timed)
    result['clip_cache_hit_rate']=hits/(hits+misses) if hits+misses else None
    result['scoring_forwards_distribution']={str(n):sum(t.get('num_option_scoring_forwards')==n for t in timed)
                                             for n in sorted({t.get('num_option_scoring_forwards',0) for t in timed})}
    result['mean_option_scoring_forwards']=mean('num_option_scoring_forwards')
    for source,target in [('PRISM_selection_latency_ms','prism_selection_ms'),
                          ('PRISM_end_to_end_latency_ms','prism_end_to_end_ms')]:
        for percentile in ('mean','median','p90','p95','p99'):
            result[percentile+'_'+target]=stats.get(source,{}).get(percentile)
    for key in ('final_generation_ms','final_ttft_ms','temporal_prefilter_ms','clip_image_encode_ms',
                'clip_text_encode_ms','clip_relevance_ms','mmr_ranking_ms','retrieval_total_ms',
                'vlm_selection_total_ms','candidate_admission_ms','arbitration_total_ms',
                'video_io_ms','full_system_latency_ms'):
        result['mean_'+key]=mean(key)
    result['mean_video_decode_ms']=distribution([(t.get('broad_history_decode_ms') or 0)+
                                  (t.get('recent6_decode_ms') or 0) for t in timed])['mean']
    for j in range(4):
        result[f'mean_k{j}_ms_executed_only']=mean(f'k{j}_total_ms')
    result['num_evaluated_states_distribution']={str(j):sum(t.get('num_sufficiency_iterations')==j for t in timed) for j in range(5)}
    result['groups']={}
    for field in ('task','duration','duration_group','subtask'):
        for group in sorted({str(r[field]) for r in rows if field in r}):
            subset=[r for r in rows if str(r.get(field))==group]
            result['groups'][field+':'+group]=dict(n=len(subset),accuracy=accuracy(subset,benchmark),
                mean_depth=distribution([profile(r)[2].get('selected_memory_depth') for r in subset])['mean'])
    if baseline:
        other=records(Path(baseline))
        b={r['pair_key']:r for r in other}
        if set(b)!={r['pair_key'] for r in rows}:
            raise ValueError('Baseline and PRISM question IDs differ')
        for r in rows:
            _,a,_=profile(r)
            _,ba,_=profile(b[r['pair_key']])
            if not ba.get('baseline_control') or ba.get('recent_frame_hashes')!=a.get('recent_frame_hashes'):
                raise ValueError('Unverified/mismatched exact Recent-6 baseline hashes')
        result['baseline_pairing']='exact keys and recent hashes verified'
        result['baseline_accuracy']=accuracy(other,benchmark)
        if result['baseline_accuracy'] is not None:
            result['accuracy_delta']=result['prism_accuracy']-result['baseline_accuracy']
        base_latency=distribution([profile(r)[2].get('PRISM_end_to_end_latency_ms') for r in other])['mean']
        ours=mean('PRISM_end_to_end_latency_ms')
        result['baseline_algorithmic_end_to_end_ms']=base_latency
        result['latency_overhead_ms']=ours-base_latency if ours is not None and base_latency else None
        result['relative_latency_overhead_pct']=100*(ours/base_latency-1) if base_latency and ours is not None else None
        delta=result['accuracy_delta']; overhead=result['latency_overhead_ms']
        result['accuracy_gain_per_100ms']=delta/(overhead/100) if delta is not None and overhead and overhead>0 else None
        pairs=[(correctness(b[r['pair_key']],benchmark),correctness(r,benchmark)) for r in rows]
        result['rescue_rate']=sum(x is False and y is True for x,y in pairs)/max(len(pairs),1) if benchmark!='streambench_v03' else None
        result['damage_rate']=sum(x is True and y is False for x,y in pairs)/max(len(pairs),1) if benchmark!='streambench_v03' else None
        baseline_ttft=distribution([profile(r)[2].get('final_ttft_ms') for r in other])['mean']
        result['ttft_overhead_pct']=100*(mean('final_ttft_ms')/baseline_ttft-1) if baseline_ttft else None
        result['frames_overhead']=(result['mean_final_frames']-distribution([profile(r)[2].get('final_frame_count') for r in other])['mean'])
        base_tokens=distribution([r.get('num_vision_tokens') for r in other])['mean']
        result['vision_token_overhead']=result['mean_vision_tokens']-base_tokens if base_tokens is not None and result['mean_vision_tokens'] is not None else None
        for name,group in result['groups'].items():
            field,value=name.split(':',1)
            pair_subset=[(correctness(b[r['pair_key']],benchmark),correctness(r,benchmark)) for r in rows if str(r.get(field))==value]
            group['rescue_rate']=sum(x is False and y is True for x,y in pair_subset)/max(len(pair_subset),1) if benchmark!='streambench_v03' else None
            group['damage_rate']=sum(x is True and y is False for x,y in pair_subset)/max(len(pair_subset),1) if benchmark!='streambench_v03' else None
    components=['temporal_prefilter_ms','clip_image_encode_ms','clip_text_encode_ms','clip_relevance_ms','mmr_ranking_ms',
        *[f'k{j}_total_ms' for j in range(4)],'decision_logic_total_ms','context_total_ms','final_generation_ms']
    e2e=mean('PRISM_end_to_end_latency_ms') or 0
    table=[]
    for k in components:
        d=distribution([t.get(k) for t in timed])
        amortized=sum(t.get(k) or 0 for t in timed)/max(len(timed),1)
        table.append(dict(component=k,**d,amortized_mean=amortized,percent_prism_e2e=100*amortized/e2e if e2e else None))
    result['largest_latency_blocks']=[r['component'] for r in sorted(table,key=lambda r:r['amortized_mean'],reverse=True)[:2]]
    result['clip_retrieval_percent']=100*(mean('retrieval_total_ms') or 0)/e2e if e2e else None
    result['minicpm_selection_percent']=100*(mean('vlm_selection_total_ms') or 0)/e2e if e2e else None
    result['final_generation_percent']=100*(mean('final_generation_ms') or 0)/e2e if e2e else None
    timing_file=directory/'job_timing.json'
    if timing_file.exists():
        wall=json.loads(timing_file.read_text())['evaluator_wall_seconds']
        result['samples_per_second_global']=len(rows)/wall if wall>0 else None
        result['throughput_scope']='Evaluator subprocess wall time including its startup; not per-question latency.'
    result['next_optimization']='Profile and optimize '+', '.join(result['largest_latency_blocks'])+' without changing accepted contexts.'
    (directory/'latency_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    scalar={k:v for k,v in result.items() if not isinstance(v,(dict,list))}
    with (directory/'latency_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(scalar));writer.writeheader();writer.writerow(scalar)
    with (directory/'latency_blocks.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    text=[DEFINITION,SYSTEM_DEFINITION,'',f'Benchmark: {benchmark}; questions: {len(rows)}; errors: {result["errors"]}',
        'Conditional averages exclude unexecuted stages; amortized averages include them as no work.',
        '| Component | Conditional mean ms | Amortized mean ms | Median | P95 | % E2E |',
        '|---|---:|---:|---:|---:|---:|']
    for r in table:
        text.append('| '+ ' | '.join(str(r.get(k)) for k in ('component','mean','amortized_mean','median','p95','percent_prism_e2e'))+' |')
    (directory/'latency_summary.md').write_text('\n'.join(text))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',action='append',required=True,help='benchmark=/absolute/result/directory; repeat for master report')
    p.add_argument('--baseline',action='append',default=[],help='benchmark=/matched/exact-control/directory')
    p.add_argument('--output',default='reports/progressive_arbitration/latency_summary')
    args=p.parse_args(); baselines=dict(x.split('=',1) for x in args.baseline)
    results=[summarize_directory(Path(d),b,baselines.get(b)) for b,d in (x.split('=',1) for x in args.run)]
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    (output/'latency_all_benchmarks.json').write_text(json.dumps(results,indent=2))
    flat=[]
    for r in results:
        item={k:v for k,v in r.items() if not isinstance(v,(dict,list))}
        item.update({k+'_mean':v['mean'] for k,v in r['metrics'].items()})
        flat.append(item)
    with (output/'latency_all_benchmarks.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in flat for k in r}));w.writeheader();w.writerows(flat)
    lines=[DEFINITION,SYSTEM_DEFINITION,'','| Benchmark | Accuracy | Baseline | Delta pp | Mean PRISM E2E ms |','|---|---:|---:|---:|---:|']
    lines += [f"| {r['benchmark_name']} | {r['prism_accuracy']} | {r['baseline_accuracy']} | {r['accuracy_delta']} | {r['metrics'].get('PRISM_end_to_end_latency_ms',{}).get('mean')} |" for r in results]
    (output/'latency_all_benchmarks.md').write_text('\n'.join(lines))

if __name__=='__main__':
    main()
