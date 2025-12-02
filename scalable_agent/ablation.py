"""
Ablation Study for Active Dreaming Memory
Compares 4 system variants as described in the research paper.
"""

from .agent import LifelongAgent
from .benchmarks.multi_domain import MultiDomainBenchmark
from .adapter import LifelongAgentAdapter
from .evaluation.statistics import StatisticalEvaluator
import json

class AblationStudy:
    def __init__(self):
        self.variants = {
            "Full System (ADM)": {"sleep": True, "symbolic": True},
            "No Sleep": {"sleep": False, "symbolic": True},
            "No Symbolic": {"sleep": True, "symbolic": False},
            "No Memory": {"sleep": False, "symbolic": False}
        }
        self.results = {}
    
    def run_variant(self, name: str, config: dict, num_tasks: int = 20) -> dict:
        """
        Run a single variant on a subset of tasks.
        """
        print(f"\n{'='*70}")
        print(f"Running Variant: {name}")
        print(f"Config: Sleep={config['sleep']}, Symbolic={config['symbolic']}")
        print(f"{'='*70}")
        
        # Create agent with specific configuration
        agent = LifelongAgent(
            enable_sleep=config['sleep'],
            enable_symbolic=config['symbolic']
        )
        agent.store.clear()  # Start fresh
        
        # Create adapter
        adapter = LifelongAgentAdapter(agent=agent)
        
        # Run on subset of benchmark
        benchmark = MultiDomainBenchmark()
        # Use first num_tasks for faster ablation
        benchmark.tasks = benchmark.tasks[:num_tasks]
        
        results = benchmark.run_evaluation(adapter)
        
        return {
            "name": name,
            "config": config,
            "success_rate": results['overall_success_rate'],
            "by_domain": results['by_domain'],
            "total_tasks": results['total_tasks']
        }
    
    def run_full_ablation(self, num_tasks: int = 30):
        """
        Run all 4 variants and compare results.
        """
        print("\n" + "="*70)
        print("ABLATION STUDY: Active Dreaming Memory")
        print("="*70)
        print(f"Testing {len(self.variants)} variants on {num_tasks} tasks each\n")
        
        all_results = []
        
        for name, config in self.variants.items():
            result = self.run_variant(name, config, num_tasks)
            all_results.append(result)
            self.results[name] = result
        
        # Print comparison table
        self._print_comparison_table(all_results)
        
        # Statistical analysis
        self._statistical_analysis(all_results)
        
        # Save results
        with open("ablation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Ablation study complete. Results saved to ablation_results.json")
        
        return self.results
    
    def _print_comparison_table(self, results: list):
        """Print formatted comparison table."""
        print("\n" + "="*70)
        print("ABLATION RESULTS")
        print("="*70)
        print(f"{'Variant':<25} {'Success Rate':<15} {'vs Baseline':<15}")
        print("-"*70)
        
        baseline_rate = next(r['success_rate'] for r in results if r['name'] == "No Memory")
        
        for r in results:
            rate = r['success_rate']
            diff = rate - baseline_rate
            diff_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
            print(f"{r['name']:<25} {rate:>6.1f}%{'':<8} {diff_str:<15}")
        
        print("="*70)
    
    def _statistical_analysis(self, results: list):
        """Perform statistical significance testing."""
        print("\n" + "="*70)
        print("STATISTICAL SIGNIFICANCE")
        print("="*70)
        
        # For demonstration, we'll show the improvement
        # In a real study, you'd run multiple trials and use actual t-tests
        full_system = next(r for r in results if r['name'] == "Full System (ADM)")
        no_memory = next(r for r in results if r['name'] == "No Memory")
        
        improvement = full_system['success_rate'] - no_memory['success_rate']
        
        print(f"Full System (ADM): {full_system['success_rate']:.1f}%")
        print(f"No Memory Baseline: {no_memory['success_rate']:.1f}%")
        print(f"Improvement: +{improvement:.1f} percentage points")
        print("\nNote: Run with multiple trials for full statistical analysis")
        print("="*70)


def run_quick_ablation():
    """Quick ablation on 10 tasks per variant."""
    study = AblationStudy()
    results = study.run_full_ablation(num_tasks=10)
    return results


def run_full_ablation():
    """Full ablation on 30 tasks per variant."""
    study = AblationStudy()
    results = study.run_full_ablation(num_tasks=30)
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        print("Running FULL ablation study (30 tasks per variant)...")
        run_full_ablation()
    else:
        print("Running QUICK ablation study (10 tasks per variant)...")
        print("Use --full flag for complete evaluation")
        run_quick_ablation()
