import time
import os
import random
import statistics
from TSP_class import GA_Engine


def run_ga_greedy():
    # 1. 파일 및 파라미터 설정
    target_filename = "testset2/cycle318.in"

    base_path = ""
    full_path = os.path.join(base_path, target_filename)

    params = {
        'MUT_RATE': 0.25,
        'POP_SIZE': 120,
        'NUM_OFFSPRING': 70,
        'TOURNAMENT_T': 0.8,
    }

    SEEDS = range(20)

    # 2. 엔진 초기화
    engine = GA_Engine(full_path, params, run_name="run_GAwithgreedy", seed=0)

    print(f"🚀 Start Standard GA (with Greedy Init) on {target_filename}")

    # Baseline: Greedy
    greedy_tour, greedy_fit = engine.greedy_solution()
    print(f"   [Baseline] Greedy Init Score: {greedy_fit:.4f}")
    engine.plot_solution(greedy_tour, f"Initial Greedy ({greedy_fit:.2f})", "plot_00_initial_greedy.png")

    summary_rows = []
    ga_results = []
    improvements = []

    print(f"{'seed':>4} | {'GA_best_len':>12} | {'improve(%)':>10} | {'Generations':>11}")
    print("-" * 52)

    for seed in SEEDS:
        random.seed(seed)
        engine.rng.seed(seed)

        # 3. 초기 집단 생성 (Greedy 포함)
        population = [(greedy_tour, greedy_fit)]
        while len(population) < params['POP_SIZE']:
            t = list(range(engine.n_cities))
            engine.rng.shuffle(t)
            population.append((t, engine.get_fitness(t)))

        population.sort(key=lambda x: x[1])

        start_time = time.time()
        trace = []
        gen = 0

        # 4. 메인 루프
        while time.time() - start_time < engine.limit_time:
            gen += 1
            offsprings = []

            for _ in range(params['NUM_OFFSPRING']):
                mom, dad = engine.selection(population)
                child = engine.crossover(mom, dad)

                if engine.rng.random() < params['MUT_RATE']:
                    child = engine.mutation(child)

                offsprings.append((child, engine.get_fitness(child)))

            population = engine.replacement(population, offsprings)

            # Trace 기록 (Gen, Time, Best, Avg)
            current_best = population[0][1]
            current_avg = sum(p[1] for p in population) / len(population)
            trace.append((gen, time.time() - start_time, current_best, current_avg))

        # 5. 결과 처리
        best_sol = population[0]
        elapsed = time.time() - start_time
        imp = (greedy_fit - best_sol[1]) / greedy_fit * 100.0

        ga_results.append(best_sol[1])
        improvements.append(imp)

        print(f"{seed:>4} | {best_sol[1]:>12.4f} | {imp:>10.2f} | {gen:>11}")

        # [저장]
        engine.save_seed_log(seed, trace)
        engine.plot_solution(best_sol[0],
                             f"Seed {seed} GA(GreedyInit) Best ({best_sol[1]:.2f})",
                             f"plot_seed_{seed}.png")
        # 확인용 기본 그래프
        engine.plot_convergence_graph(trace, seed)

        summary_rows.append([seed, best_sol[1], gen, imp, elapsed])

    # 6. 최종 통계 출력 및 저장
    mean_len = statistics.mean(ga_results)
    stdev_len = statistics.stdev(ga_results) if len(SEEDS) > 1 else 0.0

    print("-" * 52)
    print(f"[Summary] Average Fitness: {mean_len:.4f} ± {stdev_len:.4f}")

    engine.save_summary(["Seed", "Best_Fitness", "Generations", "Improvement_Pct", "Time"], summary_rows)


if __name__ == "__main__":
    run_ga_greedy()