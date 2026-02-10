import time
import random
from TSP_class import GA_Engine


def run_ga_without_greedy():
    # 1. 파일 및 파라미터 설정
    target_file = "testset2/cycle600.in"

    params = {
        'MUT_RATE': 0.25,
        'POP_SIZE': 120,
        'NUM_OFFSPRING': 70,
        'TOURNAMENT_T': 0.8,
    }

    SEEDS = range(20)

    # 2. 엔진 초기화
    engine = GA_Engine(target_file, params, run_name="run_GAwithoutgreedy", seed=0)

    print(f"🚀 Start GA without Greedy (Random Init) on {target_file}")

    # Baseline: Greedy Score
    greedy_tour, greedy_fit = engine.greedy_solution()
    print(f"   (Reference) Greedy Score: {greedy_fit:.4f}")
    engine.plot_solution(greedy_tour, f"Greedy Baseline ({greedy_fit:.2f})", "plot_00_baseline_greedy.png")

    summary_rows = []

    for seed in SEEDS:
        random.seed(seed)
        engine.rng.seed(seed)

        # 3. 초기 집단 생성
        population = []
        while len(population) < params['POP_SIZE']:
            t = list(range(engine.n_cities))
            engine.rng.shuffle(t)
            population.append((t, engine.get_fitness(t)))

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

            # Trace 기록
            current_best = population[0][1]
            trace.append((time.time() - start_time, current_best))

        # 5. 결과 처리
        best_sol = population[0]  # (tour, fit)
        elapsed = time.time() - start_time

        # 개선율
        imp = (greedy_fit - best_sol[1]) / greedy_fit * 100

        print(f"Seed {seed:>2} | Best: {best_sol[1]:.4f} | Gen: {gen} | Imp: {imp:.2f}%")

        # 6. 로그 및 시각화 저장
        engine.save_seed_log(seed, trace)
        engine.plot_solution(best_sol[0],
                             f"Seed {seed} RandomInit Best ({best_sol[1]:.2f})",
                             f"plot_seed_{seed}.png")

        summary_rows.append([seed, best_sol[1], gen, imp, elapsed])

    # 7. 최종 요약 저장
    engine.save_summary(["Seed", "Best_Fitness", "Generations", "Improvement_Pct", "Time"], summary_rows)


if __name__ == "__main__":
    run_ga_without_greedy()