"""Render predeclared report tables from saved rank/FX summaries, without refitting."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report'
RANK = ROOT / 'outputs/rank_sensitivity_analysis'
FX = ROOT / 'outputs/fx_extension_analysis'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_manifest(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    for name, expected in manifest['artifacts'].items():
        assert sha(folder / name) == expected, name
    return manifest


def one(frame, **keys):
    selected = frame
    for key, value in keys.items():
        selected = selected[selected[key] == value]
    assert len(selected) == 1, keys
    return selected.iloc[0]


def table(name, columns, headers, rows):
    end = r' \\'
    return '\n'.join(['\\newcommand{\\'+name+'}{%',r'\begin{tabular}{'+columns+'}',
        r'\toprule',*headers,r'\midrule',*[' & '.join(row)+end for row in rows],
        r'\bottomrule',r'\end{tabular}}'])


def main():
    verified_manifest(RANK)
    verified_manifest(FX)
    rank = pd.read_csv(RANK / 'reference_architecture_contrasts.csv')
    core = pd.read_csv(FX / 'report_core_table.csv')
    matrix = pd.read_csv(FX / 'report_calibration_matrix.csv')
    mechanism = pd.read_csv(FX / 'mechanism_summary.csv')
    end = r' \\'
    lines = ['% Display only. Complete rank and FX outcomes remain in saved CSVs.']
    rows = []
    for phase, label in [('entry','Entry'),('recovery64','Recovery64')]:
        rows.append([label]+[f"{one(rank, phase=phase, rule=rule)['estimate']:+.4f}"
            for rule in ('ceiling','randomized90','randomized90_minus_ceiling')])
    lines.append(table('RankReadout','lrrr',
        ['Phase & Original rank & Randomized rank & Change'+end],rows))
    rows = []
    for predictor,label in [('gru','GRU'),('tcn','TCN'),('garch11','GARCH')]:
        values = [label]
        for calibrator,window in [('native',0),('rolling',128)]:
            row = one(core,predictor=predictor,calibrator=calibrator,window=window)
            values += [f"{100*row['coverage']:.2f}"+r'\%',f"{row['normalized_interval_score']:.3f}"]
        rows.append(values)
    lines.append(table('FXMainReadout','lrrrr',[
        r'& \multicolumn{2}{c}{Native} & \multicolumn{2}{c}{Rolling128}'+end,
        'Model & Coverage & IS & Coverage & IS'+end],rows))
    rows = []
    methods = ['native','rolling32','rolling128','aci128','sfogd','saocp']
    for forecaster,label in [('constant','Constant'),('ewma16','EWMA16'),('ewma64','EWMA64'),
                              ('garch11','GARCH'),('gru64','GRU64'),('tcn64','TCN64')]:
        row = one(matrix,forecaster=forecaster)
        rows.append([label]+[f'{row[m]:.3f}' for m in methods])
    lines.append(table('FXFullReadout','lrrrrrr',[
        'Model & Native & W32 & W128 & ACI128 & SF-OGD & SAOCP'+end],rows))
    rows = []
    for architecture,label in [('gru','GRU'),('tcn','TCN'),('gru_minus_tcn','GRU$-$TCN')]:
        rows.append([label]+[f"{one(mechanism, architecture=architecture, component=c, metric='interval_score', block_length=128)['estimate']:+.4f}"
            for c in ['total','current_at_archive64','archive_at_current16']])
    lines.append(table('FXMechanismReadout','lrrr',[
        r'Model & $J_{\rm FX}$ & $C_{b64}$ & $A_{a16}$'+end],rows))
    path = OUT / 'extension_readout.tex'
    path.write_text('\n\n'.join(lines)+'\n')
    sources = [RANK/'reference_architecture_contrasts.csv',FX/'report_core_table.csv',
               FX/'report_calibration_matrix.csv',FX/'mechanism_summary.csv']
    manifest = {'operation':'Presentation only; no new bootstrap, fitting, or selection',
        'source_sha256':sha(__file__),
        'inputs':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'analysis_manifests':{str(p.relative_to(ROOT))+'/manifest.json':sha(p/'manifest.json') for p in (RANK,FX)},
        'artifacts':{str(path.relative_to(ROOT)):sha(path)},
        'scope':'Rank reference and FX report displays frozen in original extension protocols; full all-method tables retained'}
    (OUT/'extension_readout_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'rendered','statistics_changed':False}))


if __name__ == '__main__':
    main()
