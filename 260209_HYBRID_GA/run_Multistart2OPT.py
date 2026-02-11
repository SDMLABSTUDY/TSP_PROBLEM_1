import time
import os
import random
from TSP_class import TSP_Base


def run_multistart():
    cycle_files = [
        "cycle50.in",
        "cycle100.in",
        "cycle200.in",
        "cycle318.in",
        "cycle600.in"
    ]

    restarts_map = {
        "cycle50.in": 19000,  # Hybrid GA 평균값 근사치 사용
        "cycle100.in": 19000,
        "cycle200.in": 9000,
        "cycle318.in": 6500,
        "cycle600.in": 5000
    }

    SEEDS = range(3)

    for filename in cycle_files:
        target_filename = f"testset2/{filename}"

        # 파일 확인
        if not os.path.exists(target_filename):
            continue

        # 횟수 설정
        restarts = restarts_map.get(filename, 5000)

        base_path = ""
        full_path = os.path.join(base_path, target_filename)

        # 엔진 초기화
        engine = TSP_Base(full_path, run_name="run_Multistart2OPT", seed=0)
        print(f"\n🚀 Start Multi-start 2-OPT ({restarts} restarts) on {target_filename}")

        summary_rows = []

        for seed in SEEDS:
            random.seed(seed)
            engine.rng.seed(seed)

            start_time = time.time()

            best_in_seed = float('inf')
            best_tour_seed = None

            for _ in range(restarts):
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