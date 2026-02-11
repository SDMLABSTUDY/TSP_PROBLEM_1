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
                    # First Improvement 효과
                    # break

    return best_tour, engine.get_fitness(best_tour)


def run_hybrid_ga():
    # 1. 파일 리스트 설정
    cycle_files = [
        "cycle50.in",
        "cycle100.in",
        "cycle200.in",
        "cycle318.in",
        "cycle600.in"
    ]

    # 2. 파라미터 설정
    params = {
        'MUT_RATE': 0.25,
        'POP_SIZE': 120,
        'NUM_OFFSPRING': 70,
        'TOURNAMENT_T': 0.8,


        'LS_RATE': 0.3,  # 30% 확률로 자식에게 2-OPT 적용
        'LS_MAX_STEPS': 50  # 2-OPT 최대 반복 횟수
    }

    SEEDS = range(10)

    for filename in cycle_files:
        # 경로 설정
        target_filename = os.path.join("testset2", filename)

        # 파일 확인
        if not os.path.exists(target_filename):
            print(f"⚠️ Warning: {target_filename} 파일을 찾을 수 없습니다. 건너뜁니다.")
            continue

        # 엔진 초기화
        engine = GA_Engine(target_filename, params, run_name="run_hybridGA_Standard", seed=0)

        print(f"\n🚀 Start Hybrid GA [Standard Probabilistic] on {target_filename}")
        print(f"   Option: Initial Seeding + {int(params['LS_RATE'] * 100)}% LS Probability")

        # Baseline 계산 (비교용)
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

                # 초기 population 2-OPT 적용
                refined_tour, refined_fit = limited_two_opt(t, engine, max_steps=params['LS_MAX_STEPS'])
                population.append((refined_tour, refined_fit))

            population.sort(key=lambda x: x[1])

            start_time = time.time()
            trace = []
            gen = 0
            call_count_2opt = 0

            # 4. 메인 루프
            while time.time() - start_time < engine.limit_time:
                gen += 1
                offsprings = []

                for _ in range(params['NUM_OFFSPRING']):
                    # (1) 교차
                    mom, dad = engine.selection(population)
                    child = engine.crossover(mom, dad)

                    # (2) 변이
                    if engine.rng.random() < params['MUT_RATE']:
                        child = engine.mutation(child)

                    # (3) 2-OPT
                    if engine.rng.random() < params['LS_RATE']:
                        child, fit = limited_two_opt(child, engine, max_steps=params['LS_MAX_STEPS'])
                        call_count_2opt += 1

                    offsprings.append((child, engine.get_fitness(child)))

                # (4) 세대 교체
                population = engine.replacement(population, offsprings)

                # Trace 기록
                current_best = population[0][1]
                current_avg = sum(p[1] for p in population) / len(population)
                trace.append((gen, time.time() - start_time, current_best, current_avg))

            # 5. 결과 처리
            best_sol = population[0]
            elapsed = time.time() - start_time
            imp = (greedy_fit - best_sol[1]) / greedy_fit * 100.0

            print(f"{seed:>4} | {best_sol[1]:>12.4f} | {imp:>10.2f} | {gen:>8} | {call_count_2opt:>8}")

            # 로그 및 그래프 저장
            engine.save_seed_log(seed, trace)
            engine.plot_solution(best_sol[0],
                                 f"Seed {seed} Std({params['LS_RATE']}) Best ({best_sol[1]:.2f})",
                                 f"plot_seed_{seed}.png")
            engine.plot_convergence_graph(trace, seed)

            summary_rows.append([seed, best_sol[1], gen, call_count_2opt, imp, elapsed])

        # 최종 요약 저장
        engine.save_summary(["Seed", "Best_Fitness", "Generations", "2OPT_Calls", "Improvement_Pct", "Time"],
                            summary_rows)


if __name__ == "__main__":
    run_hybrid_ga()