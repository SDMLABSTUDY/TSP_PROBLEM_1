import time
from TSP_class import GA_Engine


def run_hybrid_ga():
    # 경로 확인 필수
    target_file = "testset2/cycle600.in"

    params = {
        'MUT_RATE': 0.25,
        'POP_SIZE': 120,
        'NUM_OFFSPRING': 70,
        'TOURNAMENT_T': 0.8,
    }

    SEEDS = range(20)  # 0 ~ 19

    # 1. 엔진 초기화
    engine = GA_Engine(target_file, params, run_name="run_hybridGA", seed=0)

    print(f"🚀 Start Hybrid GA (Memetic) on {target_file}")
    print(f"   Results will be saved in: {engine.result_path}")

    # 2. [시각화] 초기 Greedy 해 (Before Image) 저장
    greedy_tour, greedy_fit = engine.greedy_solution()
    engine.plot_solution(greedy_tour,
                         f"Initial Greedy (Fit: {greedy_fit:.2f})",
                         "plot_00_initial_greedy.png")
    print(f"📸 Saved Initial Greedy Plot (Fit: {greedy_fit:.2f})")

    summary_rows = []

    for seed in SEEDS:
        engine.rng.seed(seed)

        # 초기 집단 생성
        population = [(greedy_tour, greedy_fit)]
        while len(population) < params['POP_SIZE']:
            t = list(range(engine.n_cities))
            engine.rng.shuffle(t)
            population.append((t, engine.get_fitness(t)))

        start_time = time.time()
        trace = []
        gen = 0
        call_count_2opt = 0

        while time.time() - start_time < engine.limit_time:
            gen += 1
            offsprings = []

            for _ in range(params['NUM_OFFSPRING']):
                mom, dad = engine.selection(population)
                child = engine.crossover(mom, dad)

                if engine.rng.random() < params['MUT_RATE']:
                    child = engine.mutation(child)

                # Hybridization (2-OPT)
                if engine.rng.random() < params['LS_RATE']:
                    child, _ = engine.two_opt(child)
                    call_count_2opt += 1

                offsprings.append((child, engine.get_fitness(child)))

            population = engine.replacement(population, offsprings)
            trace.append((time.time() - start_time, population[0][1]))

        best_sol = population[0]
        elapsed = time.time() - start_time
        imp = (greedy_fit - best_sol[1]) / greedy_fit * 100

        print(f"Seed {seed:>2} | Best: {best_sol[1]:.4f} | Imp: {imp:.2f}%")

        # 3. [시각화] 최종 Hybrid 해 (After Image) 저장
        engine.save_seed_log(seed, trace)
        engine.plot_solution(best_sol[0],
                             f"Seed {seed} Hybrid Best ({best_sol[1]:.2f})",
                             f"plot_seed_{seed}.png")

        summary_rows.append([seed, best_sol[1], gen, call_count_2opt, imp, elapsed])

    engine.save_summary(["Seed", "Best_Fitness", "Generations", "2OPT_Calls", "Improvement_Pct", "Time"], summary_rows)


if __name__ == "__main__":
    run_hybrid_ga()