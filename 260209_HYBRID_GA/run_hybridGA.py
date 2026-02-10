import time
import os
import random
import statistics
from TSP_class import GA_Engine


def run_hybrid_ga():
    # 경로 설정
    target_filename = "testset2/cycle318.in"

    base_path = ""
    full_path = os.path.join(base_path, target_filename)

    params = {
        'MUT_RATE': 0.25,
        'POP_SIZE': 120,
        'NUM_OFFSPRING': 70,
        'TOURNAMENT_T': 0.8,
        'LS_RATE': 0.1,  # Local Search(2-OPT) 적용 확률
    }

    SEEDS = range(20)

    # 1. 엔진 초기화
    engine = GA_Engine(full_path, params, run_name="run_hybridGA", seed=0)

    print(f"🚀 Start Hybrid GA (Memetic) on {target_filename}")
    print(f"   Results will be saved in: {engine.result_path}")

    # 2. 초기 Greedy
    greedy_tour, greedy_fit = engine.greedy_solution()
    engine.plot_solution(greedy_tour,
                         f"Initial Greedy (Fit: {greedy_fit:.2f})",
                         "plot_00_initial_greedy.png")

    summary_rows = []
    ga_results = []
    improvements = []

    print(f"{'seed':>4} | {'Best':>12} | {'Imp(%)':>10} | {'2OPT_Calls':>10} | {'Gen':>8}")
    print("-" * 60)

    for seed in SEEDS:
        random.seed(seed)
        engine.rng.seed(seed)

        # 초기 집단 생성
        population = [(greedy_tour, greedy_fit)]
        while len(population) < params['POP_SIZE']:
            t = list(range(engine.n_cities))
            engine.rng.shuffle(t)
            population.append((t, engine.get_fitness(t)))

        population.sort(key=lambda x: x[1])

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

                # 1. Mutation
                if engine.rng.random() < params['MUT_RATE']:
                    child = engine.mutation(child)

                # 2. Hybridization (2-OPT)
                if engine.rng.random() < params['LS_RATE']:
                    child, fit = engine.two_opt(child)
                    call_count_2opt += 1

                offsprings.append((child, engine.get_fitness(child)))

            population = engine.replacement(population, offsprings)

            # Trace 기록
            current_best = population[0][1]
            current_avg = sum(p[1] for p in population) / len(population)
            trace.append((gen, time.time() - start_time, current_best, current_avg))

        best_sol = population[0]
        elapsed = time.time() - start_time
        imp = (greedy_fit - best_sol[1]) / greedy_fit * 100.0

        ga_results.append(best_sol[1])
        improvements.append(imp)

        print(f"{seed:>4} | {best_sol[1]:>12.4f} | {imp:>10.2f} | {call_count_2opt:>10} | {gen:>8}")

        # [저장]
        engine.save_seed_log(seed, trace)
        engine.plot_solution(best_sol[0],
                             f"Seed {seed} Hybrid Best ({best_sol[1]:.2f})",
                             f"plot_seed_{seed}.png")

        # 확인용 기본 그래프
        engine.plot_convergence_graph(trace, seed)

        summary_rows.append([seed, best_sol[1], gen, call_count_2opt, imp, elapsed])

    # 최종 요약
    engine.save_summary(["Seed", "Best_Fitness", "Generations", "2OPT_Calls", "Improvement_Pct", "Time"], summary_rows)


if __name__ == "__main__":
    run_hybrid_ga()