#!/usr/bin/env python3
"""
Aggregate all revision experiment results (exp12-exp15) into paper-ready
figures and tables. Idempotent: rerun any time; skips missing inputs.

    python analyze_revision.py --results revision_results --figdir revision_results/figures
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

COLORS = {'flat': '#1f77b4', 'hier': '#d62728', 'mean': '#7f7f7f'}


def fig_exp12(results, figdir):
    for raw in glob.glob(os.path.join(results, 'exp12_*', 'exp12_raw.csv')):
        ds = os.path.basename(os.path.dirname(raw)).replace('exp12_', '')
        df = pd.read_csv(raw)
        df = df[df.strategy == 'semi']
        if df.empty:
            continue
        g = df.groupby(['family', 'nominal_budget'])['recall@10']
        stats = g.agg(['mean', 'std', 'count']).reset_index()
        stats['ci'] = 1.96 * stats['std'] / np.sqrt(stats['count'])
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for fam, label, marker in [('flat', 'HMRC (flat k-means)', 'o'),
                                   ('hier', 'kRt-style (hierarchical)', 's')]:
            s = stats[stats.family == fam].sort_values('nominal_budget')
            ax.errorbar(s.nominal_budget, s['mean'] * 100, yerr=s.ci * 100,
                        marker=marker, capsize=3, label=label,
                        color=COLORS[fam])
        m = stats[stats.family == 'mean']
        if len(m):
            ax.axhline(m['mean'].iloc[0] * 100, ls='--', c=COLORS['mean'],
                       label='Mean centroid')
        ax.set_xlabel('Representatives per segment (budget)')
        ax.set_ylabel('Routing Recall@10 (%)')
        ax.set_title(f'{ds}: flat vs hierarchical reps.\n'
                     f'(semi-structured segments, 5 seeds, 95% CI)')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(os.path.join(figdir, f'fig_exp12_{ds}.{ext}'),
                        dpi=200)
        plt.close(fig)
        print(f'fig_exp12_{ds} written')


def fig_exp13(results, figdir):
    for raw in glob.glob(os.path.join(results, 'exp13_*', 'exp13_raw.csv')):
        ds = os.path.basename(os.path.dirname(raw)).replace('exp13_', '')
        df = pd.read_csv(raw)
        scenarios = df.scenario.unique()
        fig, axes = plt.subplots(1, len(scenarios),
                                 figsize=(4.2 * len(scenarios), 3.4),
                                 sharey=False)
        if len(scenarios) == 1:
            axes = [axes]
        for ax, sc in zip(axes, scenarios):
            sub = df[df.scenario == sc]
            lines = [('mean', 'static', 'Mean centroid', '--', COLORS['mean']),
                     ('hmrc', 'static', 'HMRC-3 (no refresh)', '-', '#ff7f0e'),
                     ('hmrc', 'drift', 'HMRC-3 (drift refresh)', '-',
                      COLORS['flat'])]
            for method, pol, label, ls, c in lines:
                s = sub[(sub.method == method) & (sub.policy == pol)]
                if s.empty:
                    continue
                g = s.groupby('step')['recall@10'].agg(['mean', 'std',
                                                        'count'])
                ci = 1.96 * g['std'] / np.sqrt(g['count'])
                ax.plot(g.index, g['mean'] * 100, ls, color=c, marker='o',
                        ms=3, label=label)
                ax.fill_between(g.index, (g['mean'] - ci) * 100,
                                (g['mean'] + ci) * 100, color=c, alpha=0.15)
            ax.set_title({'append': 'Streaming inserts',
                          'deletes': 'Skewed deletes (30%)',
                          'tenants': 'Tenant growth + drift'}.get(sc, sc))
            ax.set_xlabel('Step')
            ax.grid(alpha=0.3)
        axes[0].set_ylabel('Routing Recall@10 (%)')
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(os.path.join(figdir, f'fig_exp13_{ds}.{ext}'),
                        dpi=200)
        plt.close(fig)
        print(f'fig_exp13_{ds} written')


def fig_exp15(results, figdir):
    for raw in glob.glob(os.path.join(results, 'exp15*', 'exp15_raw.csv')):
        tag = os.path.basename(os.path.dirname(raw))
        df = pd.read_csv(raw)
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for method, c, m in [('Mean', COLORS['mean'], '^'),
                             ('HMRC-3', COLORS['flat'], 'o'),
                             ('kRt-b2d2', COLORS['hier'], 's')]:
            s = df[df.method == method]
            g = s.groupby('size')['recall@10'].agg(['mean', 'std', 'count'])
            ci = 1.96 * g['std'] / np.sqrt(g['count'])
            ax.errorbar(g.index / 1000, g['mean'] * 100, yerr=ci * 100,
                        marker=m, capsize=3, label=method, color=c)
        ax.set_xlabel('Corpus size (thousands of vectors)')
        ax.set_ylabel('Routing Recall@10 (%)')
        ax.set_title('Routing recall vs corpus scale\n'
                     '(semi-structured segments, 95% CI)')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(os.path.join(figdir, f'fig_{tag}.{ext}'), dpi=200)
        plt.close(fig)
        print(f'fig_{tag} written')


def fig_exp14(results, figdir):
    for raw in glob.glob(os.path.join(results, 'exp14_*', 'exp14_raw.csv')):
        ds = os.path.basename(os.path.dirname(raw)).replace('exp14_', '')
        df = pd.read_csv(raw)
        strategies = df.strategy.unique()
        fig, axes = plt.subplots(1, len(strategies),
                                 figsize=(4 * len(strategies), 3.4),
                                 sharey=True)
        if len(strategies) == 1:
            axes = [axes]
        width = 0.35
        for ax, st in zip(axes, strategies):
            sub = df[df.strategy == st]
            methods = ['Mean', 'HMRC-3']
            x = np.arange(len(methods))
            for off, col, label in [(-width / 2, 'real_recall@10',
                                     'Real queries'),
                                    (width / 2, 'pseudo_recall@10',
                                     'Held-out corpus vectors')]:
                vals, cis = [], []
                for meth in methods:
                    v = sub[sub.method == meth][col]
                    vals.append(v.mean() * 100)
                    cis.append(1.96 * v.std() / max(np.sqrt(len(v)), 1) * 100)
                ax.bar(x + off, vals, width, yerr=cis, capsize=3,
                       label=label)
            ax.set_xticks(x)
            ax.set_xticklabels(methods)
            ax.set_title(st)
            ax.grid(alpha=0.3, axis='y')
        axes[0].set_ylabel('Routing Recall@10 (%)')
        axes[0].legend(fontsize=8)
        fig.suptitle(f'{ds}: real vs synthetic query protocol', fontsize=11)
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(os.path.join(figdir, f'fig_exp14_{ds}.{ext}'),
                        dpi=200)
        plt.close(fig)
        print(f'fig_exp14_{ds} written')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='revision_results')
    ap.add_argument('--figdir', default=None)
    args = ap.parse_args()
    figdir = args.figdir or os.path.join(args.results, 'figures')
    os.makedirs(figdir, exist_ok=True)
    fig_exp12(args.results, figdir)
    fig_exp13(args.results, figdir)
    fig_exp14(args.results, figdir)
    fig_exp15(args.results, figdir)


if __name__ == '__main__':
    main()
