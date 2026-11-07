"""Exact saved-binary64 density moments and frozen high-precision comparisons."""
from fractions import Fraction as F
import mpmath as mp


def moments(mass, oracle, density, independent):
    m, o, u, a = [list(map(F, v.tolist())) for v in (mass, oracle, density, independent)]
    assert len(m) == len(o) == len(u) == len(a) and min(m) > 0 and min(o) > 0
    error = [x-y for x, y in zip(u, o)]
    delta = [x-y for x, y in zip(u, a)]
    dot = lambda v: sum((w*x*x for w, x in zip(m, v)), F(0))
    relative = [x/y for x, y in zip(error, o)]
    M = sum(m, F(0)); denominator = dot(o)
    return {k: str(v) for k, v in {
        'mass': M, 'oracle_square': denominator,
        'error_square': dot(error), 'evaluation_square': dot(delta),
        'signed_error_mass': sum((w*x for w, x in zip(m, error)), F(0)),
        'relative_Linf': max(map(abs, relative)),
        'evaluation_Linf': max(abs(x/y) for x, y in zip(delta, o)),
        'relative_minimum': min(relative), 'relative_maximum': max(relative),
        'density_minimum': min(u), 'density_maximum': max(u),
        'oracle_minimum': min(o), 'oracle_maximum': max(o),
        'mass_minimum': min(m), 'mass_maximum': max(m)}.items()}


def evaluate(exact, digits):
    with mp.workdps(digits):
        def value(s):
            q = F(s); return mp.mpf(q.numerator)/q.denominator
        values = {}; rows = []
        for item in exact:
            q = {k: value(v) for k, v in item['moments'].items()}
            v = {'E2': mp.sqrt(q['error_square']/q['oracle_square']),
                 'Einf': q['relative_Linf'],
                 'B2': mp.sqrt(q['evaluation_square']/q['oracle_square']),
                 'Binf': q['evaluation_Linf'],
                 'signed_mass_bias': q['signed_error_mass']/q['mass']}
            v.update({k: q[k] for k in ('relative_minimum', 'relative_maximum',
                'density_minimum', 'density_maximum', 'oracle_minimum', 'oracle_maximum',
                'mass_minimum', 'mass_maximum')})
            values[item['N'], item['source']] = v
            rows.append({'N': item['N'], 'source': item['source'],
                         **{k: mp.nstr(a, 65) for k, a in v.items()}})
        pairs = []
        for source in ('kernel', 'ledger'):
            for coarse, fine in ((64, 128), (128, 256)):
                c, f = values[coarse, source], values[fine, source]
                for metric, indicator in (('E2', 'B2'), ('Einf', 'Binf')):
                    gain = c[metric]-f[metric]; budget = c[indicator]+f[indicator]
                    decision = ('decrease_larger_than_evaluation_indicator' if gain > budget else
                        'increase_larger_than_evaluation_indicator' if gain < -budget else
                        'numerically_unresolved')
                    ratio = c[metric]/f[metric] if f[metric] else None
                    pairs.append({'source': source, 'coarse_N': coarse, 'fine_N': fine,
                        'metric': metric, 'gain': mp.nstr(gain, 65),
                        'evaluation_indicator_sum': mp.nstr(budget, 65), 'decision': decision,
                        'coarse_over_fine_error': None if ratio is None else mp.nstr(ratio, 65),
                        'observed_log2_ratio': None if ratio is None or ratio <= 0 else mp.nstr(mp.log(ratio, 2), 65)})
        decreasing = all(x['decision'] == 'decrease_larger_than_evaluation_indicator' for x in pairs)
        return {'observations': rows, 'spatial_pairs': pairs,
            'all_pairs_decrease_beyond_evaluation_indicators': decreasing,
            'decision': 'diagnostic_complete_review_only' if decreasing else 'stop_no_expansion',
            'general_order_or_PDE_bound_certified': False}
