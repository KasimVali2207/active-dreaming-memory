"""
Statistical Evaluation Module for Active Dreaming Memory
Implements paired t-tests, Cohen's d, and other metrics from the research paper.
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple
import json

class StatisticalEvaluator:
    def __init__(self):
        self.results = {}
    
    def paired_t_test(self, group1: List[float], group2: List[float], 
                      name1: str = "Group 1", name2: str = "Group 2") -> Dict:
        """
        Perform paired t-test between two groups.
        Returns t-statistic, p-value, and interpretation.
        """
        t_stat, p_value = stats.ttest_rel(group1, group2)
        
        mean_diff = np.mean(group1) - np.mean(group2)
        ci_95 = stats.t.interval(0.95, len(group1)-1, 
                                 loc=mean_diff, 
                                 scale=stats.sem(np.array(group1) - np.array(group2)))
        
        result = {
            "comparison": f"{name1} vs {name2}",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "mean_difference": float(mean_diff),
            "ci_95_lower": float(ci_95[0]),
            "ci_95_upper": float(ci_95[1]),
            "significant": p_value < 0.001,
            "significance_level": "p < 0.001" if p_value < 0.001 else f"p = {p_value:.4f}"
        }
        
        return result
    
    def cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """
        Calculate Cohen's d effect size.
        Interpretation: 0.2=small, 0.5=medium, 0.8=large, 2.0=very large
        """
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        
        d = (mean1 - mean2) / pooled_std
        return float(d)
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate comprehensive metrics from benchmark results.
        """
        total = len(results)
        successes = sum(1 for r in results if r['success'])
        success_rate = (successes / total) * 100
        
        # By domain
        by_domain = {}
        for r in results:
            domain = r['domain']
            if domain not in by_domain:
                by_domain[domain] = {"success": 0, "total": 0}
            by_domain[domain]["total"] += 1
            if r['success']:
                by_domain[domain]["success"] += 1
        
        domain_rates = {
            domain: (stats["success"] / stats["total"]) * 100
            for domain, stats in by_domain.items()
        }
        
        # By difficulty
        by_difficulty = {}
        for r in results:
            diff = r.get('difficulty', 'medium')
            if diff not in by_difficulty:
                by_difficulty[diff] = {"success": 0, "total": 0}
            by_difficulty[diff]["total"] += 1
            if r['success']:
                by_difficulty[diff]["success"] += 1
        
        return {
            "overall_success_rate": success_rate,
            "total_tasks": total,
            "successes": successes,
            "failures": total - successes,
            "by_domain": domain_rates,
            "by_difficulty": by_difficulty
        }
    
    def compare_agents(self, adm_results: List[float], baseline_results: List[float],
                       adm_name: str = "ADM", baseline_name: str = "Baseline") -> Dict:
        """
        Full statistical comparison between ADM and baseline.
        """
        # Paired t-test
        t_test = self.paired_t_test(adm_results, baseline_results, adm_name, baseline_name)
        
        # Cohen's d
        effect_size = self.cohens_d(adm_results, baseline_results)
        
        # Interpretation
        if effect_size < 0.2:
            effect_interp = "negligible"
        elif effect_size < 0.5:
            effect_interp = "small"
        elif effect_size < 0.8:
            effect_interp = "medium"
        elif effect_size < 2.0:
            effect_interp = "large"
        else:
            effect_interp = "very large"
        
        return {
            "t_test": t_test,
            "cohens_d": effect_size,
            "effect_interpretation": effect_interp,
            "adm_mean": float(np.mean(adm_results)),
            "baseline_mean": float(np.mean(baseline_results)),
            "improvement": float(np.mean(adm_results) - np.mean(baseline_results))
        }
    
    def generate_report(self, comparisons: List[Dict], output_file: str = "statistical_results.json"):
        """
        Generate and save statistical report.
        """
        report = {
            "comparisons": comparisons,
            "summary": {
                "total_comparisons": len(comparisons),
                "significant_results": sum(1 for c in comparisons if c['t_test']['significant'])
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n[Stats] Report saved to {output_file}")
        return report


if __name__ == "__main__":
    # Example usage
    evaluator = StatisticalEvaluator()
    
    # Simulated results (replace with actual benchmark data)
    adm_scores = [85, 88, 82, 90, 87, 84, 86, 89, 83, 85]
    baseline_scores = [65, 70, 62, 68, 67, 64, 66, 69, 63, 65]
    
    comparison = evaluator.compare_agents(adm_scores, baseline_scores, "ADM", "No Memory")
    
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON")
    print("="*60)
    print(f"ADM Mean: {comparison['adm_mean']:.1f}%")
    print(f"Baseline Mean: {comparison['baseline_mean']:.1f}%")
    print(f"Improvement: +{comparison['improvement']:.1f}%")
    print(f"\nt-statistic: {comparison['t_test']['t_statistic']:.2f}")
    print(f"p-value: {comparison['t_test']['significance_level']}")
    print(f"Cohen's d: {comparison['cohens_d']:.2f} ({comparison['effect_interpretation']})")
    print("="*60)
