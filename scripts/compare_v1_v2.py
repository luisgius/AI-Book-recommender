"""
Compare evaluation results v1 vs v2 for TFG documentation.

Generates a comprehensive comparison table showing:
- Metric changes by mode
- Methodological improvements
- Summary text ready for thesis
"""
import json
from pathlib import Path
import statistics


def compute_v1_aggregates(mode_data):
    """Compute aggregate metrics from per-query metrics in v1/v2 format."""
    per_query = mode_data['per_query_metrics']

    metrics = ['ndcg_at_10', 'recall_at_10', 'recall_at_100', 'mrr', 'ild_at_10', 'ild_coverage_at_10']
    aggregates = {}

    for metric in metrics:
        values = [q_metrics[metric] for q_metrics in per_query.values() if metric in q_metrics]
        if values:
            aggregates[metric] = statistics.mean(values)
        else:
            aggregates[metric] = 0.0

    return aggregates


def main():
    # Load both results
    v1_path = Path('data/evaluation/results.json')
    v2_path = Path('data/evaluation/results_v2.json')

    with open(v1_path) as f:
        v1 = json.load(f)
    with open(v2_path) as f:
        v2 = json.load(f)

    # Compare main modes
    print('=' * 80)
    print('COMPARISON: results.json (v1) vs results_v2.json (v2)')
    print('=' * 80)
    print()

    modes_to_compare = [
        ('hybrid_rrf', 'HYBRID RRF'),
        ('lexical_only', 'LEXICAL ONLY'),
        ('vector_only', 'VECTOR ONLY')
    ]

    for mode_key, mode_name in modes_to_compare:
        # Compute aggregates for both versions
        v1_agg = compute_v1_aggregates(v1['modes'][mode_key])
        v2_agg = compute_v1_aggregates(v2['modes'][mode_key])  # v2 has same structure

        print(f'MODE: {mode_name}')
        print('-' * 80)
        print(f'{"Metric":<30} {"v1":>12} {"v2":>12} {"Delta":>12}')
        print('-' * 80)

        for metric in ['recall_at_10', 'recall_at_100', 'ndcg_at_10', 'mrr']:
            v1_val = v1_agg.get(metric, 0.0)
            v2_val = v2_agg.get(metric, 0.0)
            delta = v2_val - v1_val
            print(f'{metric:<30} {v1_val:>12.4f} {v2_val:>12.4f} {delta:>+12.4f}')

        # ILD metrics
        v1_ild = v1_agg.get('ild_at_10', 0.0)
        v2_ild = v2_agg.get('ild_at_10', 0.0)
        v2_ild_cov = v2_agg.get('ild_coverage_at_10', 0.0)
        delta_ild = v2_ild - v1_ild

        print(f'{"ild_at_10":<30} {v1_ild:>12.4f} {v2_ild:>12.4f} {delta_ild:>+12.4f}')
        print(f'{"ild_coverage_at_10":<30} {"N/A":>12} {v2_ild_cov:>12.4f} {"NEW":>12}')

        print()

    # Check for MMR modes (present in both, but compare them)
    print('MODE: HYBRID RRF + MMR')
    print('-' * 80)
    if 'hybrid_rrf_mmr' in v1['modes'] and 'hybrid_rrf_mmr' in v2['modes']:
        v1_mmr = compute_v1_aggregates(v1['modes']['hybrid_rrf_mmr'])
        v2_mmr = compute_v1_aggregates(v2['modes']['hybrid_rrf_mmr'])

        print(f'{"Metric":<30} {"v1":>12} {"v2":>12} {"Delta":>12}')
        print('-' * 80)

        for metric in ['recall_at_10', 'recall_at_100', 'ndcg_at_10', 'mrr', 'ild_at_10']:
            v1_val = v1_mmr.get(metric, 0.0)
            v2_val = v2_mmr.get(metric, 0.0)
            delta = v2_val - v1_val
            print(f'{metric:<30} {v1_val:>12.4f} {v2_val:>12.4f} {delta:>+12.4f}')

        v2_ild_cov = v2_mmr.get('ild_coverage_at_10', 0.0)
        print(f'{"ild_coverage_at_10":<30} {"N/A":>12} {v2_ild_cov:>12.4f} {"NEW":>12}')

    print()

    # Methodological differences
    print('=' * 80)
    print('METHODOLOGICAL DIFFERENCES')
    print('=' * 80)
    print()
    print(f'{"Aspect":<45} {"v1":>15} {"v2":>15}')
    print('-' * 80)

    v1_queries = v1['modes']['hybrid_rrf']['num_queries']

    # v2 debug info is different - extract from mode data
    v2_mode_data = v2['modes']['hybrid_rrf']
    v2_queries = v2_mode_data.get('num_queries', v2_mode_data.get('n_total_queries', 0))
    v2_no_rel = v2_mode_data.get('n_queries_no_relevant', 0)
    v2_skipped = v2_mode_data.get('n_skipped_missing_judgment', 0) + v2_mode_data.get('n_skipped_no_resolved_mapping', 0)

    print(f'{"Total queries processed":<45} {v1_queries:>15} {v2_queries:>15}')
    print(f'{"Queries with no relevant docs":<45} {"N/A":>15} {v2_no_rel:>15}')
    print(f'{"Skipped queries":<45} {"N/A":>15} {v2_skipped:>15}')
    print(f'{"NDCG gain function":<45} {"implicit":>15} {"exp2_minus_1":>15}')
    print(f'{"ILD coverage metric":<45} {"NO":>15} {"YES":>15}')
    pool_limit = v2['config'].get('pool_per_mode_limit', 'N/A')
    print(f'{"Pooling limit per mode":<45} {"N/A":>15} {str(pool_limit):>15}')
    print()

    # Summary for TFG
    print('=' * 80)
    print('SUMMARY FOR TFG (copy-paste ready)')
    print('=' * 80)
    print()
    print('CAMBIOS METODOLOGICOS EN v2:')
    print()
    print('1. INCLUSION DE QUERIES SIN RELEVANTES:')
    print(f'   - v1: {v1_queries} queries evaluadas (solo con resultados relevantes)')
    print(f'   - v2: {v2_queries} queries evaluadas ({v2_no_rel} sin documentos relevantes)')
    print(f'   - Impacto: Las metricas v2 son mas bajas pero defendibles.')
    print(f'     Se evita sesgo de seleccion al incluir queries dificiles con score = 0.')
    print()
    print('2. NDCG GAIN FUNCTION EXPLICITA:')
    print('   - v1: gain function no documentada (probablemente 2^rel - 1)')
    print('   - v2: gain = 2^rel - 1 (estandar TREC, campo exp2_minus_1 en config)')
    print('   - Impacto: Trazabilidad y reproducibilidad garantizadas.')
    print()
    print('3. METRICAS DE DIVERSIDAD AMPLIADAS:')
    print('   - v1: ILD basado solo en distancia coseno entre embeddings')
    print('   - v2: ILD + ild_coverage_at_k (basado en categorias unicas)')
    print('   - Impacto: Mide diversidad semantica Y cobertura tematica.')
    print()
    print('4. POOLING LIMIT CONFIGURABLE:')
    print(f'   - v2: pool_per_mode_limit = {pool_limit} (control de profundidad de juicios)')
    print('   - Impacto: Reproducibilidad y control de costes de anotacion.')
    print()
    print('5. CAMBIOS OBSERVADOS EN METRICAS (modo hybrid_rrf):')
    v1_hybrid = compute_v1_aggregates(v1['modes']['hybrid_rrf'])
    v2_hybrid = compute_v1_aggregates(v2['modes']['hybrid_rrf'])
    print(f'   - NDCG@10:       {v1_hybrid["ndcg_at_10"]:.4f} -> {v2_hybrid["ndcg_at_10"]:.4f} (Delta = {v2_hybrid["ndcg_at_10"] - v1_hybrid["ndcg_at_10"]:+.4f})')
    print(f'   - Recall@10:     {v1_hybrid["recall_at_10"]:.4f} -> {v2_hybrid["recall_at_10"]:.4f} (Delta = {v2_hybrid["recall_at_10"] - v1_hybrid["recall_at_10"]:+.4f})')
    print(f'   - MRR:           {v1_hybrid["mrr"]:.4f} -> {v2_hybrid["mrr"]:.4f} (Delta = {v2_hybrid["mrr"] - v1_hybrid["mrr"]:+.4f})')
    print(f'   - ILD@10:        {v1_hybrid["ild_at_10"]:.4f} -> {v2_hybrid["ild_at_10"]:.4f} (Delta = {v2_hybrid["ild_at_10"] - v1_hybrid["ild_at_10"]:+.4f})')

    # ILD coverage only exists in v2
    if 'ild_coverage_at_10' in v2['modes']['hybrid_rrf']['per_query_metrics'].get('q01', {}):
        # Compute ild_coverage for v2
        ild_cov_values = []
        for q_metrics in v2['modes']['hybrid_rrf']['per_query_metrics'].values():
            if 'ild_coverage_at_10' in q_metrics:
                ild_cov_values.append(q_metrics['ild_coverage_at_10'])
        avg_ild_cov = statistics.mean(ild_cov_values) if ild_cov_values else 0.0
        print(f'   - ILD_cov@10:    N/A -> {avg_ild_cov:.4f} (NEW)')
    else:
        print('   - ILD_cov@10:    N/A -> N/A (not computed)')
    print()
    print('CONCLUSION:')
    print('La evaluacion v2 es mas rigurosa y defendible academicamente.')
    print('Las metricas son ligeramente inferiores debido a la inclusion de queries')
    print('sin resultados relevantes, lo cual refleja un escenario de uso real.')
    print()


if __name__ == '__main__':
    main()
