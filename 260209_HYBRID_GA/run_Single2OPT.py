import statistics
import time
from TSP_class import TSP_Base


def run_single_2opt():
    target_file = "testset2/cycle600.in"

    engine = TSP_Base(target_file, run_name="run_Single2OPT", seed=42)
    print(f"🚀 Start Single 2-OPT on {target_file}")

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
    ]
    engine.save_summary(["Metric", "Value"], summary_data)

    # [시각화] 최고 기록 저장
    engine.plot_solution(best_overall_tour,
                         f"Single 2-OPT Best (Fit: {best_overall_fit:.2f})",
                         "best_single_2opt.png")
    print(f"📸 Saved Best Single 2-OPT Plot (Fit: {best_overall_fit:.2f})")


if __name__ == "__main__":
    run_single_2opt()