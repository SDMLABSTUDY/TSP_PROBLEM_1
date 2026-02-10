import time
import os
import random
from TSP_class import TSP_Base


def run_multistart():
    target_filename = "testset2/cycle318.in"
    RESTARTS_PER_SEED = 2000  # Hybrid GA 호출 횟수와 비슷하게 설정 필요

    base_path = ""
    full_path = os.path.join(base_path, target_filename)

    SEEDS = range(20)

    engine = TSP_Base(full_path, run_name="run_Multistart2OPT", seed=0)
    print(f"🚀 Start Multi-start 2-OPT ({RESTARTS_PER_SEED} restarts)")

    summary_rows = []

    for seed in SEEDS:
        random.seed(seed)
        engine.rng.seed(seed)

        start_time = time.time()

        best_in_seed = float('inf')
        best_tour_seed = None

        for _ in range(RESTARTS_PER_SEED):
            tour = list(range(engine.n_cities))
            engine.rng.shuffle(tour)

            # 2-OPT 수행
            opt_tour, opt_fit = engine.two_opt(tour)

            if opt_fit < best_in_seed:
                best_in_seed = opt_fit
                best_tour_seed = opt_tour

        elapsed = time.time() - start_time
        print(f"Seed {seed:>2} | Best: {best_in_seed:.4f} | Time: {elapsed:.2f}s")

        # 시드별 Best 저장
        engine.plot_solution(best_tour_seed,
                             f"Seed {seed} MS-2OPT Best (Fit: {best_in_seed:.2f})",
                             f"plot_seed_{seed}.png")

        summary_rows.append([seed, best_in_seed, elapsed])

    engine.save_summary(["Seed", "Best_Fitness", "Time"], summary_rows)


if __name__ == "__main__":
    run_multistart()