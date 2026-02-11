import time
import os
import random
import statistics
from TSP_class import GA_Engine

def limited_two_opt(tour, engine, max_steps):
    best_tour = tour[:]
    improved = True
    n = engine.n_cities
    dm = engine.dist_matrix
    step_count = 0

    while improved and step_count < max_steps:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0: continue

                a, b = best_tour[i], best_tour[i + 1]
                c, d = best_tour[j], best_tour[(j + 1) % n]

                if dm[a][c] + dm[b][d] < dm[a][b] + dm[c][d]:
                    best_tour[i + 1:j + 1] = best_tour[i + 1:j + 1][::-1]
                    improved = True
                    step_count += 1

    return best_tour, engine.get_fitness(best_tour)


def run_hybrid_ga():
    cycle_files = [
        "cycle50.in",
        "cycle100.in",
        "cycle200.in",
        "cycle318.in",
        "cycle600.in"
    ]

    params = {
        'MUT_RATE': 0.25,
        'POP_SIZE': 120,
        'NUM_OFFSPRING': 70,
        'TOURNAMENT_T': 0.8,

        # 상위권 비율 조절
        'ELITE_RATE': 0.2,  # 상위 20%에게만 2-OPT 적용
        'LS_MAX_STEPS': 50  # 2-OPT 최대 반복 횟수
    }

    SEEDS = range(10)

    for filename in cycle_files:
        target_filename = f"testset2/{filename}"
        base_path = ""
        full_path = os.path.join(base_path, target_filename)

        # 1. 엔진 초기화
        engine = GA_Engine(full_path, params, run_name="run_hybridGA_EliteOnly", seed=0)

        print(f"\n🚀 Start Hybrid GA [Elite Only Strategy] on {target_filename}")
        print(f"   Option: Top {int(params['ELITE_RATE'] * 100)}% get 2-OPT, Others left alone.")

        # Baseline 계산
        greedy_tour, greedy_fit = engine.greedy_solution()

        summary_rows = []

        print(f"{'seed':>4} | {'Best':>12} | {'Imp(%)':>10} | {'Gen':>8} | {'2OPTs':>8}")
        print("-" * 60)

        for seed in SEEDS:
            random.seed(seed)
            engine.rng.seed(seed)

            # 3. 초기 집단 생성
            population = []
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

                # (1) 모든 자식 생성
                for _ in range(params['NUM_OFFSPRING']):
                    mom, dad = engine.selection(population)
                    child = engine.crossover(mom, dad)

                    if engine.rng.random() < params['MUT_RATE']:
                        child = engine.mutation(child)

                    offsprings.append((child, engine.get_fitness(child)))

                # (2) 상위권 선별
                # 성적순 정렬
                offsprings.sort(key=lambda x: x[1])

                # 상위 커트라인 계산
                elite_cutoff_idx = int(len(offsprings) * params['ELITE_RATE'])

                final_offsprings = []

                for i in range(len(offsprings)):
                    child_tour, child_fit = offsprings[i]

                    # 2-OPT 적용
                    if i < elite_cutoff_idx:
                        child_tour, child_fit = limited_two_opt(child_tour, engine, max_steps=params['LS_MAX_STEPS'])
                        call_count_2opt += 1

                    final_offsprings.append((child_tour, child_fit))

                # (3) 세대 교체
                population = engine.replacement(population, final_offsprings)

                # Trace
                current_best = population[0][1]
                current_avg = sum(p[1] for p in population) / len(population)
                trace.append((gen, time.time() - start_time, current_best, current_avg))

            # 결과 처리
            best_sol = population[0]
            elapsed = time.time() - start_time
            imp = (greedy_fit - best_sol[1]) / greedy_fit * 100.0

            print(f"{seed:>4} | {best_sol[1]:>12.4f} | {imp:>10.2f} | {gen:>8} | {call_count_2opt:>8}")

            engine.save_seed_log(seed, trace)
            engine.plot_solution(best_sol[0],
                                 f"Seed {seed} Elite({params['ELITE_RATE']}) Best ({best_sol[1]:.2f})",
                                 f"plot_seed_{seed}.png")
            engine.plot_convergence_graph(trace, seed)

            summary_rows.append([seed, best_sol[1], gen, call_count_2opt, imp, elapsed])

        engine.save_summary(["Seed", "Best_Fitness", "Generations", "2OPT_Calls", "Improvement_Pct", "Time"],
                            summary_rows)


if __name__ == "__main__":
    run_hybrid_ga()