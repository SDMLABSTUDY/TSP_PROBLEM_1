import statistics
import os
import random
from TSP_class import TSP_Base


def run_single_2opt():
    target_filename = "testset2/cycle318.in"

    base_path = ""
    full_path = os.path.join(base_path, target_filename)

    # 엔진 초기화
    engine = TSP_Base(full_path, run_name="run_Single2OPT", seed=42)
    print(f"🚀 Start Single 2-OPT on {target_filename}")

    results = []
    best_overall_tour = None
    best_overall_fit = float('inf')

    for i in range(100):
        tour = list(range(engine.n_cities))
        engine.rng.shuffle(tour)

        # 2-OPT
        opt_tour, opt_fit = engine.two_opt(tour)
        results.append(opt_fit)

        if opt_fit < best_overall_fit:
            best_overall_fit = opt_fit
            best_overall_tour = opt_tour

    # 통계 저장
    summary_data = [
        ["Total Runs", 100],
        ["Mean Fitness", statistics.mean(results)],
        ["Best (Min) Fitness", min(results)],
        ["Worst (Max) Fitness", max(results)],
        ["Std Dev", statistics.stdev(results)]
    ]
    engine.save_summary(["Metric", "Value"], summary_data)

    engine.plot_solution(best_overall_tour,
                         f"Single 2-OPT Best (Fit: {best_overall_fit:.2f})",
                         "best_single_2opt.png")

    print(f"📸 Saved Best Single 2-OPT Plot (Fit: {best_overall_fit:.2f})")
    print(f"   Mean: {statistics.mean(results):.2f}, Best: {min(results):.2f}")


if __name__ == "__main__":
    run_single_2opt()