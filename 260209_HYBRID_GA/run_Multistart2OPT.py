import time
from TSP_class import TSP_Base


def run_multistart():
    target_file = "testset2/cycle600.in"
    RESTARTS_PER_SEED = 2000
    SEEDS = range(20)

    engine = TSP_Base(target_file, run_name="run_Multistart2OPT", seed=0)
    print(f"🚀 Start Multi-start 2-OPT ({RESTARTS_PER_SEED} restarts)")

    summary_rows = []

    for seed in SEEDS:
        engine.rng.seed(seed)
        start_time = time.time()

        best_in_seed = float('inf')
        best_tour_seed = None

        for _ in range(RESTARTS_PER_SEED):
            tour = list(range(engine.n_cities))
            engine.rng.shuffle(tour)
            opt_tour, opt_fit = engine.two_opt(tour)

            if opt_fit < best_in_seed:
                best_in_seed = opt_fit
                best_tour_seed = opt_tour

        elapsed = time.time() - start_time
        print(f"Seed {seed:>2} | Best: {best_in_seed:.4f}")

        # [시각화] 시드별 Best 저장
        engine.plot_solution(best_tour_seed,
                             f"Seed {seed} MS-2OPT Best (Fit: {best_in_seed:.2f})",
                             f"plot_seed_{seed}.png")

        summary_rows.append([seed, best_in_seed, elapsed])

    engine.save_summary(["Seed", "Best_Fitness", "Time"], summary_rows)


if __name__ == "__main__":
    run_multistart()